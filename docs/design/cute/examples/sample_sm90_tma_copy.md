
## Log

[[make_tma_copy]] 中有基本的流程分析。还是有不少

```bash
Shared memory size required: 2097152 bytes
copy matrix with M: 512, N: 1024
Using SM90_TMA_LOAD and SM90_TMA_STORE with CTA_M: 128, CTA_N: 64

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy: 1316
cta_v_tile: (_128,_64):(_1@0,_1@1)
cta_t_tile: _1:_0
slayout: (_128,_64):(_64,_1)
gtensor: gmem_ptr[16b](0x76ff91000000) o (512,1024):(1024,_1)
cta_tile: (_128,_64)
cluster_size: _1

../../../include/cute/atom/copy_traits_sm90_tma.hpp, construct_tma_gbasis: 726
gtensor         : gmem_ptr[16b](0x76ff91000000) o (512,1024):(1024,_1)
slayout         : (_128,_64):(_64,_1)
cta_v_map       : (_128,_64):(_1@0,_1@1)
inv_smem_layout : (_64,_128):(_128,_1)
sidx2gmode_full : (_64,_128):(_1@1,_1@0)
smem_rank  : _2
sidx2gmode : (_64,_128):(_1@1,_1@0)
tile_gstride : (_64,_128):(_1,1024)
tma_gstride  : (_64,_128):(_1,1024)
gbasis       : (512,1024):(_1@0,_1@1)
tile_gbasis  : (_64,_128):(_1@1,_1@0)
tma_gbasis   : (_64,_128):(_1@1,_1@0)

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy_desc: 1091
gmem_tma_basis_stride : (_1@1,_1@0)
../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy_atom: 1141
gtensor: gmem_ptr[16b](0x76ff91000000) o (512,1024):(1024,_1)
slayout: (_128,_64):(_64,_1)
cta_v_map: (_128,_64):(_1@0,_1@1)
tma_gbasis: (_64,_128):(_1@1,_1@0)
smem_swizzle: Sw<0,4,3>

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy_tiled: 1204
cta_tiler : (_128,_64)
layout_v : (_64,_128):(_128,_1)
layout_V : (((_64,_128),_1)):(((_128,_1),_0))
layout_t : _1:_8192
layout_T : _1:_0
layout_TV : (_1,(((_64,_128),_1))):(_0,(((_128,_1),_0)))

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy: 1316
cta_v_tile: (_128,_64):(_1@0,_1@1)
cta_t_tile: _1:_0
slayout: (_128,_64):(_64,_1)
gtensor: gmem_ptr[16b](0x76ff91100000) o (512,1024):(1024,_1)
cta_tile: (_128,_64)
cluster_size: _1

../../../include/cute/atom/copy_traits_sm90_tma.hpp, construct_tma_gbasis: 726
gtensor         : gmem_ptr[16b](0x76ff91100000) o (512,1024):(1024,_1)
slayout         : (_128,_64):(_64,_1)
cta_v_map       : (_128,_64):(_1@0,_1@1)
inv_smem_layout : (_64,_128):(_128,_1)
sidx2gmode_full : (_64,_128):(_1@1,_1@0)
smem_rank  : _2
sidx2gmode : (_64,_128):(_1@1,_1@0)
tile_gstride : (_64,_128):(_1,1024)
tma_gstride  : (_64,_128):(_1,1024)
gbasis       : (512,1024):(_1@0,_1@1)
tile_gbasis  : (_64,_128):(_1@1,_1@0)
tma_gbasis   : (_64,_128):(_1@1,_1@0)

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy_desc: 1091
gmem_tma_basis_stride : (_1@1,_1@0)

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy_atom: 1141
gtensor: gmem_ptr[16b](0x76ff91100000) o (512,1024):(1024,_1)
slayout: (_128,_64):(_64,_1)
cta_v_map: (_128,_64):(_1@0,_1@1)
tma_gbasis: (_64,_128):(_1@1,_1@0)
smem_swizzle: Sw<0,4,3>

../../../include/cute/atom/copy_traits_sm90_tma.hpp, make_tma_copy_tiled: 1204
cta_tiler : (_128,_64)
layout_v : (_64,_128):(_128,_1)
layout_V : (((_64,_128),_1)):(((_128,_1),_0))
layout_t : _1:_8192
layout_T : _1:_0
layout_TV : (_1,(((_64,_128),_1))):(_0,(((_128,_1),_0)))

tma_load: TiledCopy
  Tiler_MN:       (_128,_64)
  TiledLayout_TV: (_1,(((_64,_128),_1))):(_0,(((_128,_1),_0)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8192):(_0,_1)
  ValLayoutDst: (_1,_8192):(_0,_1)
  ValLayoutRef: (_1,_8192):(_0,_1)
  ValueType:    16b

../../../include/cute/algorithm/copy.hpp, copy: 200
../../../include/cute/algorithm/copy.hpp, copy: 215
../../../include/cute/algorithm/copy.hpp, copy: 246
src_c: ArithTuple(0,0) o (((_64,_128),_1),_1):(((_1@0,_1@1),_0),_0)
dst_c: smem_ptr[16b](0x770000000400) o (((_64,_128),_1),_1):(((_1,_64),_0),_0)
../../../include/cute/atom/copy_atom.hpp, call: 103
../../../include/cute/atom/copy_traits_sm90_tma.hpp, copy_unpack: 78
src: ArithTuple(0,0) o (((_64,_128),_1)):(((_1@0,_1@1),_0))
src coord: (0,0)
../../../include/cute/arch/copy_sm90_tma.hpp, copy: 352
../../../include/cute/arch/copy_sm90_tma.hpp, copy: 112
tS: ArithTuple(0,0) o (((_64,_128),_1),_1,_1):(((_1@0,_1@1),_0),_0,_0)
tD: smem_ptr[16b](0x770000000400) o (((_64,_128),_1),_1,_1):(((_1,_64),_0),_0,_0)
../../../include/cute/algorithm/copy.hpp, copy: 490
../../../include/cute/algorithm/copy.hpp, copy: 200
../../../include/cute/algorithm/copy.hpp, copy: 215
../../../include/cute/algorithm/copy.hpp, copy: 246
src_c: smem_ptr[16b](0x770000000400) o (((_64,_128),_1),_1):(((_1,_64),_0),_0)
dst_c: ArithTuple(0,0) o (((_64,_128),_1),_1):(((_1@0,_1@1),_0),_0)
../../../include/cute/atom/copy_atom.hpp, call: 103
../../../include/cute/atom/copy_traits_sm90_tma.hpp, copy_unpack: 379
../../../include/cute/arch/copy_sm90_tma.hpp, copy: 1104
../../../include/cute/arch/copy_sm90_tma.hpp, copy: 1000
First 10 elements:
Input (A): 0 0.0999756 0.199951 0.300049 0.399902 0.5 0.600098 0.700195 0.799805 0.899902 
Output (B): 0 0.0999756 0.199951 0.300049 0.399902 0.5 0.600098 0.700195 0.799805 0.899902 
Verification: PASSED 
```

### tma copy Layout 是如何算出来的

```bash
copy matrix with M: 512, N: 1024
Using SM90_TMA_LOAD and SM90_TMA_STORE with CTA_M: 128, CTA_N: 64

tma_load: TiledCopy
  Tiler_MN:       (_128,_64)
  TiledLayout_TV: (_1,(((_64,_128),_1))):(_0,(((_128,_1),_0)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_8192):(_0,_1)
  ValLayoutDst: (_1,_8192):(_0,_1)
  ValLayoutRef: (_1,_8192):(_0,_1)
  ValueType:    16b
  
src_c: ArithTuple(0,0) o (((_64,_128),_1),_1):(((_1@0,_1@1),_0),_0)
```

### 如何从 layout 到 crd0， crd1 的 
```
src: ArithTuple(0,0) o (((_64,_128),_1)):(((_1@0,_1@1),_0))
src coord: (0,0)
```

### inverse 只是 transpose ？

### sidx2gmode_full 只是 inverse 的 coord tensor ？

### 查找第一个非1step 维度是为了从自然边界开始是什么意思

### sidx2gmode 是干啥的
## 背景知识
### ArithTuple 
make_tensor 有两种
- not owning 提供指针，view tensor
- owning 未提供指针，内部拥有数据，在栈上
ArithTupe 用于描述多维坐标，而不是线性索引。使得 TMA 能够吹了多维坐标。



- TMA 的限制
	- 每一个维度不超过 256

### construct_tma_gbasis
- construct_tma_gbasis 会分析这 gmem, smem 两个布局之间的关系，构建一个最优的 TMA 描述符，使得数据能高效地从 GMEM 传输到 SMEM。

1. smem 布局逆变换，找到 smme 中最大连续向量
2. 构建 smem 到 gmem 的完整映射
3. 截断不兼容部分： 找到 SMEM 和 GMEM 之间兼容的部分，避免从 GMEM 模式的中间开始
4. 构建 TMA  的gmem 基础布局
	1. 通过组合 GTensor 和映射关系得到 tile 的 GMEM 步幅
	2. 使用 coalesce_256 函数合并模式，最大合并到 256（TMA 指令的最大 box 范围）
	3. 构建最终的 TMA GMEM 基础布局，限制为最多 5 维（TMA 硬件限制）



### make_identity_layout()

make_identity_layout 是 CUTE 库中的一个函数，用于创建一个恒等布局(identity layout)。根据代码中的定义：

```cpp
template <class Shape> 
CUTE_HOST_DEVICE constexpr auto 
make_identity_layout(Shape const& shape) {   
	return make_layout(shape, make_basis_like(shape)); 
}
```

这个函数接受一个形状参数，并返回一个布局对象。其中：

- shape 是布局的形状
- `make_basis_like(shape)` 创建与形状对应的基向量布局

`make_basis_like` 函数会为每个维度创建一个基向量，例如对于形状 `(M, N)`，它会创建基向量布局 `(_1,_0):(_0,_1)`，这表示：

- 第一个维度的步幅是 `(_1, _0)` - 在第一个维度上移动1个单位，在第二个维度上移动0个单位
- 第二个维度的步幅是 `(_0, _1)` - 在第一个维度上移动0个单位，在第二个维度上移动1个单位

```bash
gbasis       : (512,1024):(_1@0,_1@1)
```


### recast\<TmaInternalType>
 
recast\<TmaInternalType> 为何会改变 layout：因为数据类型位宽不一样

