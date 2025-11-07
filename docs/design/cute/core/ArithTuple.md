# CuTe 中的 ArithTuple 文档

## 1. 概述

ArithTuple（算术元组）是 CuTe 库中的一个重要概念，它扩展了标准的 `tuple` 类型，提供了数学运算能力。ArithTuple 主要用于表示多维坐标，并支持加法、减法和乘法等运算。

## 2. 核心组件

### 2.1 ArithmeticTuple

`ArithmeticTuple` 是一个继承自 `tuple` 的结构体，提供了数学运算能力：

```cpp
template <class... T>
struct ArithmeticTuple : tuple<T...>
{
  // 构造函数
  template <class... U>
  constexpr ArithmeticTuple(ArithmeticTuple<U...> const& u);
  
  template <class... U>
  constexpr ArithmeticTuple(tuple<U...> const& u);
  
  template <class... U>
  constexpr ArithmeticTuple(U const&... u);
};
```

主要特性：
- 支持元组间的加法和减法运算
- 支持元组的取负运算
- 支持与标量的特殊运算（如与0的加法）

### 2.2 ArithmeticTupleIterator

`ArithmeticTupleIterator` 是一个包装器，包含一个 ArithmeticTuple 作为坐标：

```cpp
template <class ArithTuple>
struct ArithmeticTupleIterator
{
  using value_type   = ArithTuple;
  using element_type = ArithTuple;
  using reference    = ArithTuple;

  ArithTuple coord_;

  constexpr ArithmeticTupleIterator(ArithTuple const& coord = {});
  constexpr ArithTuple operator*() const;
  template <class Coord>
  constexpr auto operator+(Coord const& c) const;
};
```

### 2.3 ScaledBasis

`ScaledBasis<T,N>` 表示一个至少 N+1 维的 ArithmeticTuple，其中第 N 个位置是值 T，其他位置是 0：

```cpp
template <class T, int N>
struct ScaledBasis : private tuple<T>
{
  constexpr ScaledBasis(T const& t = {});
  constexpr decltype(auto) value();
  constexpr decltype(auto) value() const;
  constexpr static auto mode();
};
```

### 2.4 E 模板别名

[E](file:///home/luyao/workspace/cutlass/cfx-article-src/external/cutlass/tools/library/include/cutlass/library/library.h#L444-L444) 是 `ScaledBasis` 的快捷方式：

```cpp
// E<>    := _1
// E<0>   := (_1,_0,_0,...)
// E<1>   := (_0,_1,_0,...)
// E<0,0> := ((_1,_0,_0,...),_0,_0,...)
// E<0,1> := ((_0,_1,_0,...),_0,_0,...)
// E<1,0> := (_0,(_1,_0,_0,...),_0,...)
// E<1,1> := (_0,(_0,_1,_0,...),_0,...)
template <int... N>
using E = typename detail::Basis<N...>::type;
```

## 3. 主要用途

### 3.1 TMA Tensors

TMA（Tensor Memory Accelerator）是 ArithTuple 最重要的应用场景。TMA 指令需要多维坐标而不是线性索引，ArithTuple 提供了处理这种坐标的有效方式。

```cpp
// 创建 TMA 坐标迭代器
ArithmeticTupleIterator citer_1 = make_inttuple_iter(42, Int<2>{}, Int<7>{});
ArithmeticTupleIterator citer_2 = citer_1 + make_tuple(Int<0>{}, 5, Int<2>{});
// 输出: (42,7,_9)
```

### 3.2 Identity Tensors

Identity tensors 使用 ArithTuple 来创建将坐标映射到自身的张量：

```cpp
template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_identity_tensor(Shape const& shape)
{
  return make_coord_tensor(make_identity_layout(shape));
}
```

### 3.3 Coordinate Tensors

Coordinate tensors 使用 ArithTuple 来表示坐标系统：

```cpp
template <class Layout>
CUTE_HOST_DEVICE constexpr
auto
make_coord_tensor(Layout const& layout)
{
  return make_tensor(make_inttuple_iter(coprofile(layout)), layout);
}
```

## 4. 工具函数

### 4.1 make_arithmetic_tuple

创建 ArithmeticTuple 实例：

```cpp
template <class... T>
CUTE_HOST_DEVICE constexpr
auto
make_arithmetic_tuple(T const&... t) {
  return ArithmeticTuple<T...>(t...);
}
```

### 4.2 make_inttuple_iter

创建 ArithmeticTupleIterator 实例：

```cpp
template <class Tuple>
CUTE_HOST_DEVICE constexpr
auto
make_inttuple_iter(Tuple const& t) {
  return ArithmeticTupleIterator(as_arithmetic_tuple(t));
}
```

### 4.3 make_basis_like

为给定形状创建类似基的结构：

```cpp
template <class Shape>
CUTE_HOST_DEVICE constexpr
auto
make_basis_like(Shape const& shape)
{
  if constexpr (is_integral<Shape>::value) {
    return Int<1>{};
  } else {
    // 为每个维度生成基
    return transform(tuple_seq<Shape>{}, shape, [](auto I, auto si) {
      // 为 si 的每个等级生成基并在前面添加 i
      using I_type = decltype(I);
      return transform_leaf(make_basis_like(si), [](auto e) {
        constexpr int i = I_type::value;
        return ScaledBasis<decltype(e), i>{};
      });
    });
  }
}
```

## 5. 运算符重载

### 5.1 加法和减法

```cpp
// 元组间运算
template <class... T, class... U>
constexpr auto operator+(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u);

template <class... T, class... U>
constexpr auto operator-(ArithmeticTuple<T...> const& t, ArithmeticTuple<U...> const& u);

// 取负
template <class... T>
constexpr auto operator-(ArithmeticTuple<T...> const& t);
```

### 5.2 乘法

ScaledBasis 支持与标量的乘法：

```cpp
template <class A, class T, int N>
constexpr auto operator*(A const& a, ScaledBasis<T,N> const& e);

template <class T, int N, class B>
constexpr auto operator*(ScaledBasis<T,N> const& e, B const& b);
```

## 6. 打印和显示

ArithTuple 提供了专门的打印函数：

```cpp
template <class ArithTuple>
CUTE_HOST_DEVICE void print(ArithmeticTupleIterator<ArithTuple> const& iter)
{
  printf("ArithTuple"); print(iter.coord_);
}

template <class T, int N>
CUTE_HOST_DEVICE void print(ScaledBasis<T,N> const& e)
{
  print(e.value()); printf("@%d", N);
}
```

## 7. 实际应用示例

### 7.1 在 GEMM 中的使用

```cpp
// 创建坐标张量用于边界检查
Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));

// 在 epilogue 中使用
auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
Tensor tCcD = thr_mma.partition_C(cD);
```

### 7.2 在 TMA 操作中的使用

```cpp
// 创建 TMA 坐标张量
auto tensor_multimode = make_tensor(ArithmeticTupleIterator(lower_corner), gemm_shapes, gbasis_strides);
```

## 8. 总结

ArithTuple 是 CuTe 库中一个强大的工具，它扩展了标准元组的功能，提供了数学运算能力。虽然它在 TMA tensors 中最为常见，但它在许多其他场景中也有重要应用，如 identity tensors、coordinate tensors 和各种需要多维坐标运算的场景。通过提供统一的接口来处理多维坐标和代数运算，ArithTuple 使得 CuTe 能够高效地处理复杂的张量操作。