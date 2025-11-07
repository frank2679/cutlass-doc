## Rank & Mode
rank：维度数
mode：具体某个维度的 idx
## group_modes 函数功能

`group_modes<Start, End>(tensor)` 的作用是将张量中从 `Start` 到 `End-1` 的模式合并为单个模式，形成一个新的张量布局。

### 具体功能：

1. **模式合并**：将指定范围内的多个模式合并成一个单一模式
2. **保持数据不变**：不改变张量中的实际数据，只改变其布局描述
3. **重新组织视图**：提供一种不同的方式来看待和访问相同的数据

### 示例说明：

假设我们有一个4D张量 `sA`，其形状为 `(4, 8, 16, 32)`：

```cpp
auto sA = make_tensor(..., make_layout(make_shape(_4{}, _8{}, _16{}, _32{})));

// 原始张量有4个模式:
// mode 0: size=4
// mode 1: size=8
// mode 2: size=16
// mode 3: size=32

// 使用 group_modes<0,2> 合并前两个模式 (0和1)
auto grouped = group_modes<0,2>(sA);
// 结果张量形状: (32, 16, 32)
// mode 0: size=32 (原来mode 0和1的组合: 4*8=32)
// mode 1: size=16 (原来的mode 2)
// mode 2: size=32 (原来的mode 3)

// 使用 group_modes<1,3> 合并中间两个模式 (1和2)
auto grouped2 = group_modes<1,3>(sA);
// 结果张量形状: (4, 128, 32)
// mode 0: size=4  (原来的mode 0)
// mode 1: size=128 (原来mode 1和2的组合: 8*16=128)
// mode 2: size=32 (原来的mode 3)

// 使用 group_modes<0,4> 合并所有模式
auto flattened = group_modes<0,4>(sA);
// 结果张量形状: (16384)
// mode 0: size=16384 (所有模式的组合: 4*8*16*32=16384)
```

### 在你的代码中的应用：

在 `group_modes<0,2>(sA)` 中：
- 合并了 sA 张量的第0和第1模式（mode 0 和 mode 1）
- 结果是一个新张量，其第一个模式是原来前两个模式的组合
- 其余模式依次向前移动

### 实际用途：

这种操作在高性能计算中非常有用：

1. **内存访问优化**：重新组织数据布局以适应特定的内存访问模式
2. **计算优化**：将多维索引简化为较少维度的索引计算
3. **硬件适配**：适配特定硬件（如Tensor Core）的数据布局要求
4. **算法简化**：将复杂的多维操作转换为简单的低维操作

### 与相关API的关系：

- `group_modes<0, Rank>(tensor)` 相当于将张量完全展平
- `group_modes<0, 1>(tensor)` 不改变张量（只合并第一个模式）
- `ungroup_modes` 是相反的操作，用于将单个模式分解为多个模式

这种模式重组是CUTE库强大功能的体现，允许开发者灵活地操作数据布局以优化性能，同时保持代码的清晰性和可维护性。