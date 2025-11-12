# CuTe TMA Copy 操作详解

TMA (Tensor Memory Access) 是 NVIDIA Hopper 架构引入的一种新的内存访问技术。它允许线程块(cluster)直接从全局内存加载数据到共享内存，或者从共享内存存储数据到全局内存，而无需显式地通过每个线程进行复制。

## TMA 基本概念

TMA 操作通过专用硬件单元执行，可以直接在全局内存和共享内存之间传输数据。这与传统的由 CUDA 线程执行的内存复制操作不同，TMA 操作由硬件管理，可以实现更高的带宽利用率和更低的寄存器压力。

### TMA 的优势

TMA 操作具有以下优势：

1. **更高的带宽利用率**：TMA 操作可以利用更高的内存带宽，比传统的线程复制更高效。
2. **减少寄存器压力**：数据直接从全局内存传输到共享内存，避免了中间的寄存器存储。
3. **硬件加速**：TMA 操作由专用硬件执行，减轻了 CUDA 核心的负担。
4. **自动向量化**：TMA 可以自动进行向量化内存访问，提高效率。
5. **多播支持**：TMA 支持将数据同时传输到多个线程块，进一步提高效率。

## TMA 操作类型

CuTe 提供了几种 TMA 操作：

### SM90_TMA_LOAD

从全局内存加载数据到共享内存。这是最基本的 TMA 加载操作。

定义在 `cute/arch/copy_sm90_tma.hpp` 中：

```cpp
struct SM90_TMA_LOAD
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, ...);
};
```

### SM90_TMA_LOAD_MULTICAST

从全局内存加载数据到共享内存，并广播到多个线程块。

定义在 `cute/arch/copy_sm90_tma.hpp` 中：

```cpp
struct SM90_TMA_LOAD_MULTICAST
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, ...);
};
```

### SM90_TMA_STORE

从共享内存存储数据到全局内存。

定义在 `cute/arch/copy_sm90_tma.hpp` 中：

```cpp
struct SM90_TMA_STORE
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr,
       void const* smem_ptr,
       int32_t const& crd0, int32_t const& crd1, ...);
};
```

### SM90_TMA_REDUCE_ADD

从共享内存归约加法数据到全局内存。

定义在 `cute/arch/copy_sm90_tma.hpp` 中：

```cpp
struct SM90_TMA_REDUCE_ADD
{
  CUTE_HOST_DEVICE static void
  copy(void const* const desc_ptr,
       void const* const smem_ptr,
       int32_t const& crd0, int32_t const& crd1, ...);
};
```

## TMA 相关数据结构

### TmaDescriptor

TMA 操作需要一个 TMA Descriptor 来描述内存访问的属性。这个描述符包含了数据的形状、步幅、内存布局等信息。

TmaDescriptor 是一个不透明的结构体，其具体实现未公开，但包含以下关键信息：

```cpp
// TMA Descriptor 结构体（概念性表示）
struct TmaDescriptor {
  // 全局内存地址
  // void* gmem_address;
  
  // TMA 张量的维度（最多5维）
  // uint64_t global_dim[5];
  
  // 全局内存步幅（以字节为单位）
  // uint64_t global_stride[4]; // 第一个步幅隐式为元素大小
  
  // 共享内存 box 尺寸
  // uint32_t smem_box_dim[5];
  
  // 共享内存元素步幅
  // uint32_t smem_elem_stride[5];
  
  // 数据类型信息
  // CUtensorMapDataType data_type;
  
  // TMA 交换模式
  // CUtensorMapSwizzle smem_swizzle;
  
  // 其他 TMA 配置参数
  // ...
};
```

TMA Descriptor 包含的关键信息有：

1. **全局内存地址**：指向要访问的全局内存起始地址
2. **张量维度信息**：描述要访问的张量在各个维度上的大小（最多支持5维）
3. **全局内存步幅**：描述在全局内存中各个维度之间的字节步幅
4. **共享内存 box 尺寸**：描述每次 TMA 操作在共享内存中访问的数据块大小
5. **共享内存元素步幅**：描述在共享内存中元素的布局方式
6. **数据类型信息**：描述张量元素的数据类型（如 float、half 等）
7. **内存交换模式**：描述共享内存的交换（swizzle）模式，用于优化内存访问
8. **其他配置参数**：包括缓存提示、交错模式等

TMA Descriptor 通过 CUDA 驱动 API 函数 `cuTensorMapEncodeTiled` 创建，该函数会根据提供的参数填充描述符结构。

### AuxTmaParams

辅助 TMA 参数结构，包含构建 TMA Descriptor 所需的一些中间参数信息。这些参数在创建 TMA Descriptor 时会被使用，但在最终的 TMA Descriptor 中可能不会直接可见。

定义在 `cute/atom/copy_traits_sm90_tma.hpp` 中：

```cpp
template <class GmemTmaBasisStrides_, class TmaGmemBasis_, class TmaSwizzle_>
struct AuxTmaParams {
  using GmemStrides  = GmemTmaBasisStrides_;    // Strides for Gmem mode -> Tma coord mode, may be dynamic
  GmemStrides g_stride_;
  using TmaGmemBasis = TmaGmemBasis_;           // Layout for Tma box shape -> Gmem mode(s), always static
  static_assert(is_static<TmaGmemBasis>::value);
  using TmaSwizzle   = TmaSwizzle_;             // Tma swizzle, always Swizzle<B,M,S>
  static_assert(is_static<TmaSwizzle>::value);
};
```

AuxTmaParams 中的参数含义：

1. **GmemStrides (g_stride_)**：
   - 类型：GmemTmaBasisStrides_
   - 含义：全局内存模式到 TMA 坐标模式的步幅映射
   - 作用：描述全局内存张量的各个维度如何映射到 TMA 操作的坐标系统
   - 可能是动态的，因为全局内存张量的步幅可能是运行时确定的

2. **TmaGmemBasis**：
   - 类型：TmaGmemBasis_
   - 含义：TMA box 形状到全局内存模式的布局映射
   - 作用：描述 TMA 操作的 box 尺寸如何映射到全局内存的各个维度
   - 始终是静态的，因为 TMA box 的形状在编译时就确定了

3. **TmaSwizzle**：
   - 类型：TmaSwizzle_
   - 含义：TMA 交换模式
   - 作用：描述共享内存的交换模式，用于优化内存访问
   - 始终是静态的，因为交换模式在编译时就确定了

AuxTmaParams 与 TMA Descriptor 的关系：

1. AuxTmaParams 包含的是构建 TMA Descriptor 所需的中间参数
2. TMA Descriptor 是最终传递给硬件的不透明结构体
3. 在 `make_tma_copy` 函数中，AuxTmaParams 的信息会被用来创建 TMA Descriptor
4. 虽然两者包含的信息有重叠，但 AuxTmaParams 更多是 CuTe 内部使用的中间表示，而 TMA Descriptor 是传递给 CUDA 驱动 API 的最终表示

## 使用 TMA 的基本流程

使用 TMA 操作的基本流程如下：

1. **创建 TMA Descriptor**：使用 `make_tma_copy` 或相关函数创建 TMA Copy 对象，该对象内部会创建 TMA Descriptor。
2. **初始化 mbarrier**：使用 mbarrier 同步 TMA 操作，确保操作完成后再访问数据。
3. **执行 TMA 操作**：调用 `copy` 函数执行 TMA 加载或存储。
4. **同步**：使用适当的同步机制等待 TMA 操作完成。

## TMA Copy 示例

以下是一个使用 TMA 加载数据的完整示例：

```cpp
// 定义全局内存张量
Tensor gmem_tensor = make_tensor(make_gmem_ptr<float>(gmem_ptr), 
                                 make_shape(M, N), 
                                 GenRowMajor{});

// 定义共享内存布局
auto smem_layout = tile_to_shape(GMMA::Layout_MN_SW128_Atom<float>{}, 
                                 make_shape(BLK_M, BLK_N));

// 创建 TMA Copy 对象
auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

// 获取 TMA tensor
auto tma_tensor = tma_load.get_tma_tensor(make_shape(M, N));

// 分割数据块
auto tma_gmem = local_tile(tma_tensor, make_shape(BLK_M, BLK_N), blk_coord);

// 获取线程片
auto thr_x = tma_load.get_slice(thread_idx);

// 分区源和目标张量
auto tma_gmem_x = thr_x.partition_S(tma_gmem);    // 分区全局内存张量
auto tma_smem_x = thr_x.partition_D(smem_tensor); // 分区共享内存张量

// 执行 TMA 加载
copy(tma_load.with(barrier, mcast_mask), tma_gmem_x, tma_smem_x);
```

## TMA 相关函数和类

### make_tma_copy

创建 TMA Copy 对象的主要函数。

定义在 `cute/atom/copy_traits_sm90_tma.hpp` 中：

```cpp
template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size);
```

### Copy_Traits 特化

针对不同 TMA 操作的 Copy_Traits 特化，定义了操作的属性。

例如，SM90_TMA_LOAD 的特化：

```cpp
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  using RefLayout = SrcLayout;

  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;
  
  // ... 其他成员函数
};
```

## TMA 多播操作

TMA 支持多播操作，可以将数据同时传输到多个线程块。

### create_tma_multicast_mask

创建 TMA 多播掩码的函数：

```cpp
template <class CtaLayout, class CtaCoord>
CUTE_HOST_DEVICE constexpr
uint16_t
create_tma_multicast_mask(CtaLayout const& cta_layout_vmnk,
                          CtaCoord  const& cta_coord_vmnk);
```

## TMA 同步机制

TMA 操作是异步的，需要适当的同步机制：

### tma_store_fence

为后续的 TMA_STORE 操作设置共享内存存储的 fence：

```cpp
CUTE_HOST_DEVICE static void
tma_store_fence();
```

### tma_store_wait

等待 TMA 操作完成：

```cpp
template <int Count>
CUTE_HOST_DEVICE static void
tma_store_wait();
```

## 注意事项

使用 TMA 操作时需要注意以下事项：

1. **硬件要求**：TMA 操作仅在 NVIDIA Hopper 架构 (SM90) 及以上版本中可用。
2. **内存对齐**：TMA 操作对内存对齐有严格要求，需要确保数据按要求对齐。
3. **共享内存布局**：需要使用特定的共享内存布局以获得最佳性能。
4. **同步**：必须正确使用同步机制，确保 TMA 操作完成后再访问相关数据。