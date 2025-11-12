# CuTe Layout 布局系统

Layout 是 CuTe 的核心概念之一，它描述了逻辑坐标到线性内存位置的映射关系。通过 Layout，CuTe 能够实现复杂的数据重排和内存访问模式。

## Layout 基本概念

Layout 定义了从多维逻辑坐标到一维线性位置（通常以位或字节为单位）的映射。它由两个主要部分组成：

- Shape（形状）：描述每个维度的大小
- Stride（步幅）：描述每个维度的跨度

### Layout 的数学表示

Layout 可以表示为一个函数：

```
L(c) = sum(c[i] * stride[i]) for i in range(rank)
```

其中 c 是逻辑坐标，stride 是步幅向量，L(c) 是线性偏移量。

## Layout 的创建

Layout 可以通过多种方式创建：

### 基本 Layout 创建

```cpp
// 创建一个 2D 行主序 Layout
auto layout_2d = make_layout(make_shape(3, 4), GenRowMajor{});
// 等价于 make_layout(make_shape(3, 4), make_stride(4, 1));

// 创建一个 2D 列主序 Layout
auto layout_2d_col = make_layout(make_shape(3, 4), GenColMajor{});
// 等价于 make_layout(make_shape(3, 4), make_stride(1, 3));
```

### Layout 的组合

Layout 可以通过多种方式组合，创建更复杂的内存访问模式：

```cpp
// 组合两个 Layout
auto layout_combined = make_layout(make_shape(Shape<_2,_3>{}, Shape<_4,_5>{}),
                                   make_stride(Stride<_1,_6>{}, Stride<_2,_7>{}));
```

## Layout 操作

CuTe 提供了丰富的 Layout 操作函数，用于创建、转换和组合 Layout。

### 基本操作

- **composition**：组合两个 Layout
- **complement**：计算 Layout 的补集
- **compact**：创建紧凑版本的 Layout
- **coalesce**：合并连续的维度
- **flatten**：展平 Layout

### Layout 组合示例

```cpp
// 创建两个 Layout
auto layout_a = make_layout(make_shape(2, 3), make_stride(1, 2));
auto layout_b = make_layout(make_shape(6, 1), make_stride(1, 0));

// 组合 Layout
auto composed_layout = composition(layout_a, layout_b);
```

## Layout 在 Tensor 中的应用

Layout 与 Tensor 紧密结合，定义了 Tensor 数据在内存中的存储方式。

```cpp
// 创建一个带有特定 Layout 的 Tensor
auto tensor = make_tensor(make_gmem_ptr<float>(ptr), 
                         make_layout(make_shape(16, 32), GenRowMajor{}));
```

## Layout 的类型

CuTe 提供了多种预定义的 Layout 类型：

### 基本 Layout 类型

- **Layout**：基本的 Layout 类型，由 Shape 和 Stride 组成，用于描述任意维度的数据布局
- **LogicalLayout**：逻辑 Layout，用于描述逻辑坐标空间的布局，不直接关联物理内存地址
- **PhysicalLayout**：物理 Layout，用于描述物理内存中的实际布局，与具体的内存地址相关联

### 特殊 Layout

- **GenRowMajor**：生成行主序 Layout，创建按行优先顺序排列的内存布局
- **GenColMajor**：生成列主序 Layout，创建按列优先顺序排列的内存布局
- **Swizzle**：内存交换 Layout，用于在共享内存中实现特定的内存访问模式以优化性能

## Layout 的属性

Layout 具有多种属性，可以通过函数获取：

### 基本属性

- **rank**：Layout 的维度数
- **size**：Layout 覆盖的元素总数
- **cosize**：Layout 的共大小

### 示例

```cpp
auto layout = make_layout(make_shape(3, 4), make_stride(4, 1));

// 获取 Layout 属性
auto layout_rank = rank(layout);  // 返回 2
auto layout_size = size(layout);  // 返回 12
```

## Layout 的转换

Layout 可以进行多种转换操作：

### 常见转换

- **recast**：重新解释 Layout 的元素类型
- **right_inverse**：计算 Layout 的右逆
- **left_inverse**：计算 Layout 的左逆
- **zip**：压缩多个 Layout

### Layout 转换示例

```cpp
// 创建一个 Layout
auto layout = make_layout(make_shape(4, 4), make_stride(1, 4));

// 计算右逆
auto inv_layout = right_inverse(layout);
```

## Layout 的实际应用

Layout 在实际应用中发挥着重要作用：

### 内存访问优化

通过合理设计 Layout，可以优化内存访问模式，提高缓存命中率和内存带宽利用率。

### 数据重排

Layout 可以实现复杂的数据重排操作，如矩阵转置、分块等。

### 硬件适配

Layout 可以适配不同的硬件特性，如向量化内存访问、共享内存交换等。

## Layout 与 Copy 操作

Layout 在 Copy 操作中起着关键作用，它定义了源和目标数据的内存布局。

```cpp
// 定义源 Layout 和目标 Layout
auto src_layout = make_layout(make_shape(16, 16), GenRowMajor{});
auto dst_layout = make_layout(make_shape(16, 16), GenColMajor{});

// 创建 Tensors
auto src_tensor = make_tensor(src_ptr, src_layout);
auto dst_tensor = make_tensor(dst_ptr, dst_layout);

// 执行 Copy 操作（包含转置）
copy(src_tensor, dst_tensor);
```

## Layout 与 MMA 操作

在 MMA (Matrix Multiply-Accumulate) 操作中，Layout 定义了矩阵元素在寄存器和内存中的布局。

```cpp
// 定义 MMA 操作相关的 Layout
using MMA_LAYOUT = Layout<Shape<_2,_2>, Stride<_1,_2>>;

// 使用 Layout 创建 MMA 操作
auto mma_layout = MMA_LAYOUT{};
```

Layout 是 CuTe 系统中的核心抽象之一，通过它实现了对复杂内存访问模式的灵活控制，为高性能计算提供了基础支持。