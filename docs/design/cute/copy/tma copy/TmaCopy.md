
H100 引入的新硬件 TMA 来提升数据搬运的效率，TMA 需要一个 desc 来描述要做的数据搬运的任务。具体参考： [[TmaDesc]]

## TMA Copy 实现架构

### 1. 核心组件

#### TMA CopyOperation
```cpp
struct SM90_TMA_LOAD {};           // TMA加载操作
struct SM90_TMA_LOAD_MULTICAST {}; // TMA多播加载操作
struct SM90_TMA_STORE {};          // TMA存储操作
struct SM90_TMA_REDUCE_ADD {};     // TMA归约加法操作
```
- SM90_TMA_LOAD 中包含 SM90_TMA_LOAD_1D/2D/3D 等
- 参数包括：
	- desc: 硬件 buffer，包含
	- barrier
	- cache_hint
	- smem_ptr: src 首地址
	- coord: 指定从哪里开始搬

```cpp
struct SM90_TMA_LOAD_1D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "l"(cache_hint)
      : "memory");
#endif
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use tma without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
```
#### Copy_Traits模板特化
针对每种TMA操作，都有对应的`Copy_Traits`特化实现，例如：
```cpp
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
```

具体定义如下：
```cpp
// The non-executable SM90_TMA_LOAD with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {&tma_desc_, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  with(
    TmaDescriptor const* new_tma_desc,
    uint64_t& tma_mbar,
    [[maybe_unused]] uint16_t const& multicast_mask = 0,
    TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {new_tma_desc, &tma_mbar, static_cast<uint64_t>(cache_hint)};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable SM90_TMA_LOAD with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_OP, NumBitsPerTMA>
  : TMA_LOAD_Unpack<SM90_TMA_LOAD_OP, NumBitsPerTMA>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint64_t   // cache hint
  > const opargs_;

  CUTE_HOST_DEVICE
  Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar, uint64_t cache)
    : opargs_(desc, mbar, cache) {}
};

```

具体可参考：[[CopyTraits 设计模式]]
### 2. TMA Descriptor构建流程

#### 主要函数：`make_tma_copy`
```cpp
template <class TmaInternalType, class CopyOp, ...>
auto make_tma_copy(CopyOp const& copy_op,
                   Tensor<GEngine,GLayout> const& gtensor,
                   SLayout const& slayout,
                   CTA_Tiler const& cta_tiler,
                   Cluster_Size const& cluster_size)
```

#### 构建步骤：
1. **构建TMA基础** (`construct_tma_gbasis`)：
   - 分析GMEM和SMEM布局关系
   - 找到最大的连续向量以优化TMA box大小
   - 构建GMEM到TMA坐标的映射

2. **创建TMA描述符** (`make_tma_copy_desc`)：
   - 设置GMEM地址、形状和步长
   - 设置SMEM box形状和步长
   - 调用CUDA API (`cuTensorMapEncodeTiled`) 创建TMA描述符

3. **创建Copy_Atom**：
   - 包装TMA描述符和辅助参数
   - 返回可执行的`Copy_Atom`对象

### 3. 执行机制

#### Copy_Atom执行流程
```cpp
// 在Copy_Atom::call中调用
copy_unpack(static_cast<Traits const&>(*this), src, dst);
```

#### TMA_LOAD_Unpack模板
```cpp
template <class CopyOp, class... Args>
struct TMA_LOAD_Unpack {
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits<CopyOp, Args...> const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout> & dst) {
    // 调用实际的TMA指令
    detail::explode_tuple(detail::CallCOPY<CopyOp>{}, ...);
  }
}
```

#### 底层TMA指令调用
在[copy_sm90_tma.hpp](file:///home/luyao/workspace/cutlass/cutlass/include/cute/arch/copy_sm90_tma.hpp)中定义了具体的汇编指令调用：
```cpp
struct SM90_TMA_LOAD_1D {
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void* smem_ptr, int32_t const& crd0) {
    asm volatile (
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3}], [%2], %4;"
      : : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "r"(crd0), "l"(cache_hint)
      : "memory");
  }
}
```

### 4. 多播支持

对于[SM90_TMA_LOAD_MULTICAST](file:///home/luyao/workspace/cutlass/cutlass/include/cute/arch/copy_sm90_tma.hpp#L770-L809)：
- 通过`multicast_mask`参数指定参与的CTA
- 使用不同的汇编指令：`multicast::cluster`

### 5. 内存屏障和同步

- 使用`mbarrier`进行同步
- 提供`TMA::CacheHintSm90`缓存提示
- 支持[prefetch](file:///home/luyao/workspace/cutlass/cutlass/include/cute/arch/copy.hpp#L98-L104)操作进行预取

### 6. IM2COL模式支持

- 支持图像到列的转换操作
- 专门的[SM90_TMA_LOAD_IM2COL](file:///home/luyao/workspace/cutlass/cutlass/include/cute/arch/copy_sm90_tma.hpp#L556-L619)操作
- 处理卷积等需要im2col转换的场景

### 7. 分区和切片

通过[TiledCopy](file:///home/luyao/workspace/cutlass/cutlass/include/cute/atom/copy_atom.hpp#L189-L206)和`ThrCopy`实现线程级分区：
```cpp
auto cta_tma = tma.get_slice(cta_idx_in_cluster);
Tensor tAgA = cta_tma.partition_S(gA);  // 分区源张量
Tensor tAsA = cta_tma.partition_D(sA);  // 分区目标张量
```

## 总结

TMA Copy的实现采用分层架构：
1. **高层接口**：`make_tma_copy`系列函数提供用户友好的API
2. **中间层**：`Copy_Atom`和`TiledCopy`封装TMA操作和分区逻辑
3. **底层实现**：通过特化的`Copy_Traits`和汇编指令直接调用硬件TMA功能

这种设计既保证了性能，又提供了足够的灵活性来适应不同的内存布局和使用场景。