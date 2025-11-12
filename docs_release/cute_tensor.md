# CuTe Tensor 张量操作

Tensor 是 CuTe 中的核心数据结构，它结合了数据指针和 Layout，提供了对多维数据的高效访问和操作。

## Tensor 基本概念

Tensor 由两个主要组件构成：

1. **Engine**：管理数据的存储和访问
2. **Layout**：定义逻辑坐标到线性内存位置的映射

### Tensor 的数学表示

Tensor 可以表示为：

```
T(c) = E(L(c))
```

其中：
- T(c) 是 Tensor 在坐标 c 处的值
- E 是 Engine，负责数据访问
- L(c) 是 Layout，将逻辑坐标 c 映射到线性位置

## Tensor 的创建

Tensor 可以通过多种方式创建，适应不同的内存空间和使用场景。

### 基本 Tensor 创建

```cpp
// 创建全局内存 Tensor
auto gmem_ptr = make_gmem_ptr<float>(ptr);
auto gmem_tensor = make_tensor(gmem_ptr, make_layout(make_shape(128, 64), GenRowMajor{}));

// 创建共享内存 Tensor
extern __shared__ float smem[];
auto smem_ptr = make_smem_ptr<float>(smem);
auto smem_tensor = make_tensor(smem_ptr, make_layout(make_shape(32, 32), GenRowMajor{}));

// 创建寄存器 Tensor
float reg_data[8];
auto reg_tensor = make_tensor(reg_data, make_layout(make_shape(2, 4), GenRowMajor{}));
```

### 不同内存空间的 Tensor

1. **全局内存 Tensor**：使用 `make_gmem_ptr` 创建
2. **共享内存 Tensor**：使用 `make_smem_ptr` 创建
3. **寄存器 Tensor**：直接使用数组创建

## Tensor 操作

CuTe 提供了丰富的 Tensor 操作函数。

### 基本访问操作

```cpp
// 通过坐标访问元素
auto value = tensor(0, 1);  // 访问 (0,1) 位置的元素

// 通过线性索引访问元素
auto value = tensor(5);     // 访问线性索引为 5 的元素

// 获取子张量
auto sub_tensor = tensor(_, 2);  // 获取第二列的所有元素
```

### Tensor 变换操作

1. **切片操作**：提取 Tensor 的一部分
2. **重塑操作**：改变 Tensor 的形状
3. **转置操作**：交换 Tensor 的维度

### Tensor 变换示例

```cpp
// 创建一个 4x4 Tensor
auto tensor_4x4 = make_tensor(ptr, make_layout(make_shape(4, 4), GenRowMajor{}));

// 切片操作：获取前两行
auto top_half = tensor_4x4(make_range(0, 2), _);

// 重塑操作：将 4x4 重塑为 2x8
auto tensor_2x8 = reshape(tensor_4x4, make_layout(make_shape(2, 8)));

// 转置操作
auto tensor_4x4_t = transpose(tensor_4x4);
```

## Tensor 与 Layout 的关系

Tensor 的行为很大程度上由其 Layout 决定。

### Layout 对 Tensor 的影响

1. **内存访问模式**：Layout 决定了如何将逻辑坐标映射到物理内存地址
2. **数据重排**：通过不同的 Layout 实现数据的重排
3. **性能优化**：合理的 Layout 设计可以优化内存访问性能

### Layout 影响示例

```cpp
// 行主序 Layout
auto row_major_tensor = make_tensor(ptr, make_layout(make_shape(4, 4), GenRowMajor{}));
// 访问模式：(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)...

// 列主序 Layout
auto col_major_tensor = make_tensor(ptr, make_layout(make_shape(4, 4), GenColMajor{}));
// 访问模式：(0,0), (1,0), (2,0), (3,0), (0,1), (1,1)...
```

## Tensor 在 Copy 操作中的应用

Tensor 在 Copy 操作中起着核心作用，定义了源和目标数据的结构。

### Copy 操作示例

```cpp
// 定义源和目标 Tensor
auto src_tensor = make_tensor(src_ptr, make_layout(make_shape(16, 16), GenRowMajor{}));
auto dst_tensor = make_tensor(dst_ptr, make_layout(make_shape(16, 16), GenColMajor{}));

// 执行 Copy 操作（包含转置）
copy(src_tensor, dst_tensor);
```

## Tensor 在 MMA 操作中的应用

在 MMA (Matrix Multiply-Accumulate) 操作中，Tensor 用于表示矩阵的输入和输出。

### MMA 操作示例

```cpp
// 定义 MMA 操作的输入 Tensor
auto A_tensor = make_tensor(A_ptr, make_layout(make_shape(8, 4), GenRowMajor{}));
auto B_tensor = make_tensor(B_ptr, make_layout(make_shape(8, 4), GenColMajor{}));
auto C_tensor = make_tensor(C_ptr, make_layout(make_shape(8, 8), GenRowMajor{}));

// 执行 MMA 操作
mma(A_tensor, B_tensor, C_tensor);
```

## 高级 Tensor 操作

CuTe 提供了一些高级 Tensor 操作，用于复杂场景。

### 张量分块

```cpp
// 创建大张量
auto big_tensor = make_tensor(ptr, make_layout(make_shape(128, 128), GenRowMajor{}));

// 分块操作
auto tile = local_tile(big_tensor, make_shape(32, 32), make_coord(1, 2));
// 获取坐标 (1,2) 处的 32x32 子块
```

### 张量组合

```cpp
// 组合多个张量
auto combined_tensor = make_tensor(ptr, make_layout(
    make_shape(Shape<_2,_3>{}, Shape<_4,_5>{}),
    make_stride(Stride<_1,_6>{}, Stride<_2,_7>{})));
```

## Tensor 的性能考虑

在使用 Tensor 时，需要考虑以下性能因素：

### 1. 内存访问模式

合理的 Layout 设计可以优化内存访问模式，提高缓存命中率。

### 2. 数据对齐

确保 Tensor 数据在内存中正确对齐，以获得最佳性能。

### 3. 向量化访问

通过合适的 Layout 实现向量化内存访问。

### 性能优化示例

```cpp
// 优化前：非对齐访问
auto slow_tensor = make_tensor(ptr+1, make_layout(make_shape(16, 16), GenRowMajor{}));

// 优化后：对齐访问
auto fast_tensor = make_tensor(ptr, make_layout(make_shape(16, 16), GenRowMajor{}));
```

## Tensor 与线程协作

在多线程环境中，Tensor 可以与线程协作实现并行计算。

### 线程切片

```cpp
// 创建 TiledCopy
auto tiled_copy = make_tiled_copy(CopyAtom{}, make_shape(32, 32), make_shape(4, 8));

// 获取线程切片
auto thread_slice = tiled_copy.get_slice(thread_idx);

// 分区张量
auto src_frag = thread_slice.partition_S(src_tensor);
auto dst_frag = thread_slice.partition_D(dst_tensor);
```

Tensor 是 CuTe 系统中的核心抽象，通过与 Layout 和 Engine 的结合，提供了对多维数据的强大操作能力。理解 Tensor 的工作机制对于高效使用 CuTe 至关重要。