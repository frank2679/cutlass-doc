# CuTe WGmma SM90 (Hopper) 详解

在 NVIDIA Hopper (SM90) 架构中，引入了新一代的 GMMA (Group Matrix Multiply-Accumulate) 指令，这些指令提供了更高的性能和更灵活的内存布局支持。本文档将详细介绍 SM90 架构下的 GMMA 相关概念、数据结构和使用方法。

## GMMA 概述

GMMA 是 Hopper 架构中引入的新一代矩阵乘法累加指令，与之前架构中的 MMA 指令相比，GMMA 提供了以下改进：

- 支持更大的矩阵操作（如 64x128x16）
- warpgroup 级别的协作（128个线程）
- 更灵活的共享内存布局支持
- 支持多种数据类型组合

## GmmaDescriptor 结构

在使用共享内存的 GMMA 操作中，[fma](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/arch/mma_sm90_gmma.hpp#L424-L430) 函数的 `desc_a` 和 `desc_b` 参数使用 [GmmaDescriptor](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/arch/mma_sm90_desc.hpp#L58-L87) 类型，该结构包含了 swizzle 信息和其他内存布局参数：

```cpp
union GmmaDescriptor
{
  // ... 其他成员 ...

  // Bitfield implementation avoids the need for shifts in assignment
  struct {
    // start_address, bit [0,14), 4LSB not included
    uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused
    // leading dimension byte offset, bit [16,30), 4LSB not included
    // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
    //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
    // For T: This is the stride from the first 8 rows to the next 8 rows.
    uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused
    // stride dimension byte offset, bit [32,46), 4LSB not included
    // For N: This is the stride from the first 8 rows to the next 8 rows.
    // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
    uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused
    // base_offset, bit [49,52)
    // Valid only for SWIZZLE_128B and SWIZZLE_64B
    uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused
    // layout type, bit [62,64)
    // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
    uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
  } bitfield;

  // ... 其他成员 ...
};
```

其中 `layout_type_` 字段直接描述了 swizzle 类型：
- LayoutType::INTERLEAVE (0): 无 swizzle
- LayoutType::B128 (1): 128字节 swizzle
- LayoutType::B64 (2): 64字节 swizzle
- LayoutType::B32 (3): 32字节 swizzle

## Swizzle 信息的获取

[make_gmma_desc](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L204-L251) 函数通过分析张量的布局信息来确定 swizzle 类型。具体来说，它使用 [layout_type](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L125-L146) 函数来提取 swizzle 信息：

```cpp
template <class Engine, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
LayoutType
layout_type(Tensor<Engine, Layout<Shape,Stride>> const&)
{
  static_assert(is_same<uint128_t, typename Engine::value_type>::value,
                "Expected uint128_t type in LayoutType conversion.");

  using Swizzle = get_swizzle_t<Engine>;
  constexpr int B = Swizzle::num_bits;
  constexpr int M = Swizzle::num_base;
  constexpr int S = Swizzle::num_shft;

  static_assert(M == 4,           "Unsupported layout swizzle");
  static_assert(0 <= B && B <= 3, "Unsupported layout swizzle");
  static_assert(S == 3,           "Unsupported layout swizzle");

  switch (B) {
    case 0: return LayoutType::INTERLEAVE;
    case 1: return LayoutType::B32;
    case 2: return LayoutType::B64;
    case 3: return LayoutType::B128;
  }
  return LayoutType::INTERLEAVE;  // ERROR
}
```

这个函数从张量的 Engine 类型中提取 Swizzle 信息，然后根据 Swizzle 的 [num_bits](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L51-L51) 字段确定 swizzle 类型：
- 当 [num_bits](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L51-L51) = 0 时，对应 LayoutType::INTERLEAVE (无 swizzle)
- 当 [num_bits](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L51-L51) = 1 时，对应 LayoutType::B32 (32字节 swizzle)
- 当 [num_bits](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L51-L51) = 2 时，对应 LayoutType::B64 (64字节 swizzle)
- 当 [num_bits](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L51-L51) = 3 时，对应 LayoutType::B128 (128字节 swizzle)

## GMMA Descriptor 的构建过程

GMMA Descriptor 是通过 [make_gmma_desc](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L204-L251) 函数构建的。这个函数接受一个共享内存张量作为参数，并根据张量的布局信息创建相应的描述符：

```cpp
template <Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
GmmaDescriptor
make_gmma_desc(Tensor<TEngine,TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "GMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "GMMA Descriptors can only be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;
  
  // ... 实现细节 ...
}
```

这个函数会分析张量的内存布局，提取起始地址、步幅等信息，并根据布局类型设置相应的 swizzle 模式。

## DescriptorIterator 和 smem_desc

在 CuTe 中，[DescriptorIterator](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L288-L313) 是一个包装 [GmmaDescriptor](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/arch/mma_sm90_desc.hpp#L58-L87) 的迭代器，用于在 GMMA 操作中传递描述符信息。[smem_desc](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L315-L364) 是一个模板结构体，用于创建共享内存描述符张量：

```cpp
template <Major>
struct smem_desc : DescriptorIterator {};
```

通过 [MakeTensor](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L362-L374) 定制点创建 [smem_desc](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L315-L364) 张量：

```cpp
// Customization point for creating a GMMA::smem_desc Tensor
template <SM90::GMMA::Major MajorMode>
struct MakeTensor<SMMA::GMMA::smem_desc<MajorMode>>
{
  template <class TEngine, class TLayout>
  CUTE_HOST_DEVICE constexpr auto
  operator()(Tensor<TEngine,TLayout> const& smem_tensor)
  {
    static_assert(is_smem<TEngine>::value, "Expected SMEM Tensor to construct a GMMA Desc Tensor");
    return make_tensor(SM90::GMMA::DescriptorIterator{SM90::GMMA::make_gmma_desc<MajorMode>(tensor<0>(smem_tensor))},
                       replace<0>(recast<uint128_t const>(smem_tensor).layout(), Layout<_1,_0>{}));
  }
};
```

## GMMA 内存布局

GMMA 支持多种共享内存布局，包括 M|N-major 和 K-major 布局：

### M|N-major 布局

```cpp
// M|N-major GMMA layouts in units of bits
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;
```

### K-major 布局

```cpp
// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
```

## GMMA 操作示例

典型的 GMMA 操作使用示例：

```cpp
// 定义输入张量 A、B 和累加张量 C
auto A_tensor = make_tensor(A_ptr, A_layout);
auto B_tensor = make_tensor(B_ptr, B_layout);
auto C_tensor = make_tensor(C_ptr, C_layout);

// 创建 GMMA 操作对象
auto mma_atom = MMA_Atom<SM90_64x64x16_F32F16F16F32_NT>{};

// 获取 MMA 操作的参与者
auto mma_thr = mma_atom.get_thread_slice(thread_idx);

// 创建累加片段
auto accum_fragment = make_fragment_like(C_tensor);
clear(accum_fragment);

// 执行 MMA 操作
mma_thr.call(A_tensor, B_tensor, accum_fragment);
```

## 总结

SM90 架构中的 GMMA 指令提供了强大的矩阵运算能力，通过 [GmmaDescriptor](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/arch/mma_sm90_desc.hpp#L58-L87) 和相关的布局工具，开发者可以高效地利用共享内存进行大规模矩阵运算。理解这些概念和工具对于充分利用 Hopper 架构的计算能力至关重要。