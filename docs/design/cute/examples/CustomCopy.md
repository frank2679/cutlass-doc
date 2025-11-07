数据类型体现在 CopyAtom ，不在 copyOperation 上

## Todo
- [x] 修改现有 demo
	- [x] 使用 make_custom_copy
	- [x] 参数化 copy numBits
	- [x] 接口修改，使用 desc，ptr, crd
		- [x] Traits tma_traits{tma_desc, aux_params}; 走的什么构造接口？见 [[CopyTraits 设计模式]]
		- [x] 创建了 desc，并且能够打通整个流程
		- [x] 计算 coord，为啥不直接传入 glayout，而不是 gshape。我直接用 glayout 
		- [x] 完善 desc 的内容，模拟实现 copy operation 底层实现，完成功能
	- [x] 支持 gmem 是 dynamic layout 的场景
	- [x] 实现更大 shape 的 Copy，依然是一维支持，M 大一点没问题，如果 N 也大一点就不行，因为底层 copy 是一维的，目前用 local tile 获取到的 coord 是一维的 offset。
		- [x] algorithm/copy 需要重新实现，当前绕过，未来可能
		- [x] copy_atom::call 需要重新实现，当前绕过
		- [x] copy 中对 coord，stride 的使用不对，当前绕过，在两维场景中解决
	- [x] 支持两维 operation
		- [x] coord, arithmatic tensor 操作两维 coord，使用 identity tensor，local_tile 只要传入的是 tensor 就可以
		- [x] 增加支持 2D，兼容原来的 1D 场景
		- [x] 增加 1D 用例，用来验证 1D 用例的正确性。
	- [x] 尝试 TiledCopy
		- [x] 是什么：make_tma_copy_tiled 是一个底层实现，返回 TiledCopy
		- [ ] 能工作，但是具体的实现没看懂，或者还没找到这个 tiled 的作用
		- [ ] tma_partition 做啥的 [[tma_partition]]
			- [x] tile_to_shape(tile, shape) [[tile_to_shape]]
			- [x] 为何说 right_inverse 是找到最连续的部分，逆布局的功能是什么，如何实现的[[right_inverse]]
	- [x] copy，copy_atom 的处理，复用 or 重载，复用需要修改维度，需要继续套壳，将内部的维度包起来，变成一个 rank，要不就要重载这两个接口，但是似乎不是很好重载。
		- [x] 参考田源他们的实现进行重载
	- [ ] 如果 grid 为 1，是否现在的版本就不行了。对的，需要内部自己 loop。这也是 general mma 的场景。
- [x] 改写 general mma copy
	- [x] 迁移现有 demo 到 T chip
	- [x] 将现有 demo 支持 tma-like api copy
- [x] 哪里做了 bit to sizeof(T) 的转换？哪里用得上 bits ? 
- [ ] Multicast 如何实现
- [ ] 同步信号是否作为 operation 参数


### prompt
参考 tma copy ，修改 customCopy 的接口，接口参数为 desc, smem_ptr, crd0，虽然我可能还是用类似 Universal copy 来实现底层copy，可以先不用管 barrier 和  cache_hint 参数。重点是在 make_custom_copy 中构造 desc，然后使用 desc 来传参。

### C++
- [x] todo with is not used why 导致了找不到特化的实现？是的


## 设计
### CopyOperation

参数：
- desc
	- gmem_ptr
	- layout_src
	- layout_dst
	- datatype
	- swizzle
	- oobFill
- smem_ptr
- coord: 用来计算 gmem 的 offset

### CopyTraits

属性
- numOfbits


## NumOfBits

对齐到 bits 的好处，以及哪里用到它
- 硬件友好，fp32, int8 等都是 bit 为单位，尤其未来还有 int4 等类型
- CopyTraits 中的 layout 用到，所有 traits 都是 NumOfBitsPerTMA
- copy_atom 中的 NumVal 读取到 Traits 中的 layout
- partition 中用到 copy_atom 中的 NumVal
- 总而言之，memory 大小都用 bit 为单位来描述。


## 带 coord 的 tensor

1. src_coord = tma_load.get_tma_tensor(shape)
2. scr_coord_cta = local_tile(src_coord)
3. load_per_cta = tma_load.get_slice(tid)
4. t_src = load_per_cta.partition_S(src_coord_cta)
5. t_dst = load_per_cta.partition_D(smem_tensor)
6. copy(tma_load.with(barrier), t_src, t_dst)


## algorithm copy & copy atom
algorithm 层会将 2D copy 拆成多个 loop ，但是我们其实不用。
不过可以看一下 tma copy 是否有拆成多个 loop。


## 常用操作

### 公共的

#### local_tile

### tiledCopy/tiledMMA
#### partition

#### get_slice

