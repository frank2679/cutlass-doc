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

### LayoutType 枚举

LayoutType 定义了不同的内存布局类型：

- `INTERLEAVE`: 无 swizzle 操作
- `B32`: 32 字节 swizzle
- `B64`: 64 字节 swizzle
- `B128`: 128 字节 swizzle

## Swizzle 机制详解

### Swizzle 参数 (B, M, S)

在 CuTe 中，Swizzle 是一个模板类，定义如下：

```cpp
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
```

三个模板参数的含义：

1. **BBits (B)**: 表示参与 swizzle 操作的位数，即掩码中的位数
2. **MBase (M)**: 表示保持不变的最低有效位数
3. **SShift (S)**: 表示 YYY 掩码的移位距离（正数表示向右移位，负数表示向左移位）

### Swizzle 工作原理

Swizzle 操作的位布局如下：

```
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase 是保持不变的最低有效位数
                 ^-^       ^-^     BBits 是掩码中的位数
                   ^---------^     SShift 是 YYY 掩码的移位距离
```

例如：

```
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
```

结果是：

```
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx 其中 AA = ZZ xor YY
```

### B32, B64, B128 数值含义

B32、B64 和 B128 这三个 swizzle 模式的数值代表了不同的 swizzle 粒度和内存访问模式：

1. **B32 (数值1)**: 32字节 swizzle 模式
   - 使用 `Swizzle<1,4,3>` 模板参数
   - 对 32 字节 (256 位) 的数据块进行 swizzle 操作
   - 适用于较小的数据块或特定的内存访问模式

2. **B64 (数值2)**: 64字节 swizzle 模式
   - 使用 `Swizzle<2,4,3>` 模板参数
   - 对 64 字节 (512 位) 的数据块进行 swizzle 操作
   - 提供中等粒度的内存访问优化

3. **B128 (数值3)**: 128字节 swizzle 模式
   - 使用 `Swizzle<3,4,3>` 模板参数
   - 对 128 字节 (1024 位) 的数据块进行 swizzle 操作
   - 提供最大粒度的内存访问优化，适用于大规模数据处理

这些数值在 [layout_type](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp#L125-L146) 函数中被映射为对应的 LayoutType 枚举值：

- `num_bits = 1` 对应 `LayoutType::B32` (32字节 swizzle)
- `num_bits = 2` 对应 `LayoutType::B64` (64字节 swizzle)
- `num_bits = 3` 对应 `LayoutType::B128` (128字节 swizzle)

数值越大表示 swizzle 操作的粒度越大，可以更好地优化大规模数据的内存访问模式，但也可能增加实现的复杂性。

### Layout 如何描述 Swizzle

在 CuTe 中，Layout 可以通过 [ComposedLayout](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/layout_composed.hpp#L51-L63) 来描述 swizzle。[ComposedLayout](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/layout_composed.hpp#L51-L63) 是一种组合布局，它将多个布局或变换组合在一起，其中就包括 Swizzle 变换。

一个典型的 swizzle layout 定义如下：

```cpp
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape< _1024,_8>,Stride<_1,_1024>>>;
```

这个定义包含三个部分：

1. **Swizzle<3,4,3>**: Swizzle 变换函数，描述了如何进行内存地址的 swizzle 操作
2. **smem_ptr_flag**: 偏移量，通常为 0，用于调整地址偏移
3. **Layout<Shape< _1024,_8>,Stride<_1,_1024>>**: 基础布局，描述了数据的基本排列方式

通过这种组合方式，Layout 能够完整地描述包含 swizzle 变换的复杂内存布局。当需要创建实际的张量时，CuTe 会将这些信息综合起来，生成能够正确访问 swizzled 内存的代码。

例如，在 GMMA 操作中，常见的 swizzle layouts 定义如下：

```cpp
// M|N-major GMMA layouts in units of bits
using Layout_MN_INTER_Atom_Bits = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape< _128,_8>,Stride<_1, _128>>>;
using Layout_MN_SW32_Atom_Bits  = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape< _256,_8>,Stride<_1, _256>>>;
using Layout_MN_SW64_Atom_Bits  = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape< _512,_8>,Stride<_1, _512>>>;
using Layout_MN_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_1024,_8>,Stride<_1,_1024>>>;

// K-major GMMA layouts in units of bits
using Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
using Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
using Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
using Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
```

这些定义展示了如何使用 [ComposedLayout](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/layout_composed.hpp#L51-L63) 将 Swizzle 变换与基础内存布局组合起来，形成完整的内存访问模式描述。

### Swizzle 机制实现原理

Swizzle 机制通过位操作来重新排列内存访问模式，以提高内存带宽利用率和缓存局部性。在 CuTe 中，Swizzle 的实现基于位掩码操作：

```cpp
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle
{
  static constexpr int num_bits = BBits;
  static constexpr int num_base = MBase;
  static constexpr int num_shft = SShift;

  // 使用 'int' 类型以避免无意中转换为无符号数
  using bit_msk = cute::constant<int, (1 << num_bits) - 1>;
  using yyy_msk = cute::constant<int, bit_msk{} << (num_base + max(0,num_shft))>;
  using zzz_msk = cute::constant<int, bit_msk{} << (num_base - min(0,num_shft))>;
  using msk_sft = cute::constant<int, num_shft>;

  static constexpr uint32_t swizzle_code = uint32_t(yyy_msk::value | zzz_msk::value);

  template <class Offset>
  CUTE_HOST_DEVICE constexpr static
  auto
  apply(Offset const& offset)
  {
    return offset ^ shiftr(offset & yyy_msk{}, msk_sft{});   // ZZZ ^= YYY
  }

  template <class Offset>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(Offset const& offset) const
  {
    return apply(offset);
  }
};
```

Swizzle 的核心实现原理是通过位操作来交换特定位置的位：

1. **位掩码创建**：
   - `bit_msk` 创建一个包含 [num_bits](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L51-L51) 个1的掩码
   - `yyy_msk` 创建 YYY 位的掩码，位置根据 [num_base](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L52-L52) 和 [num_shft](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L53-L53) 确定
   - `zzz_msk` 创建 ZZZ 位的掩码，位置同样根据 [num_base](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L52-L52) 和 [num_shft](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L53-L53) 确定

2. **位操作应用**：
   - 通过 `offset & yyy_msk{}` 提取 YYY 位
   - 使用 `shiftr` 函数根据 [msk_sft](file:///Users/joycezhao/workspace/cutlass-doc/cutlass/include/cute/swizzle.hpp#L56-L56) 移位
   - 最后通过异或操作 (`^`) 将移位后的 YYY 位与 ZZZ 位交换

这种位操作机制允许在编译时确定内存访问模式，从而优化 GPU 内存子系统的使用。

### 如何从 Tensor Engine 中提取 BMS 参数

`layout_type` 函数通过 `get_swizzle_t` 从 Tensor 的 Engine 中提取 Swizzle 信息：

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

对应关系：

- `num_bits = 0` 对应 `LayoutType::INTERLEAVE` (无 swizzle)
- `num_bits = 1` 对应 `LayoutType::B32` (32字节 swizzle)
- `num_bits = 2` 对应 `LayoutType::B64` (64字节 swizzle)
- `num_bits = 3` 对应 `LayoutType::B128` (128字节 swizzle)

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