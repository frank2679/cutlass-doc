# CuTe MMA (Matrix Multiply-Accumulate) 操作

MMA (Matrix Multiply-Accumulate) 是 CuTe 中用于执行矩阵乘加运算的核心组件。在深度学习和科学计算中，MMA 操作是最关键的计算之一。

## MMA 基本概念

MMA 操作执行以下计算：

```cpp
D = A * B + C
```

其中 A、B 是输入矩阵，C 是累加矩阵，D 是输出矩阵。

在 CuTe 中，MMA 操作被高度抽象化，允许开发者：

- 使用不同精度的数据类型
- 利用专门的硬件指令（如 Tensor Cores）
- 适应不同的内存布局
- 与线程协作进行大规模计算

## CuTe MMA 抽象层次

CuTe 对 MMA 操作进行了多层抽象，从底层硬件指令到高级接口：

### 1. Operation 结构体

Operation 结构体封装了特定的 PTX 指令。它定义了指令所需的寄存器类型和实际的 fma 函数实现。

设计原因：

- 将底层硬件指令封装在统一接口下，隐藏硬件差异
- 明确定义寄存器使用模式，便于编译器优化
- 提供类型安全的接口，防止寄存器类型错误

例如，`SM70_8x8x4_F32F16F16F32_NT` 定义了 Volta 架构上的一个 MMA 操作：

```cpp
struct SM70_8x8x4_F32F16F16F32_NT
{
  // 定义 D 矩阵使用的寄存器类型和数量
  // 8个float寄存器用于存储8x8矩阵的输出结果
  using DRegisters = float[8];
  
  // 定义 A 矩阵使用的寄存器类型和数量
  // 2个uint32_t寄存器，每个包含2个F16值（共4个F16值）
  using ARegisters = uint32_t[2];
  
  // 定义 B 矩阵使用的寄存器类型和数量
  // 与A矩阵相同，2个uint32_t寄存器
  using BRegisters = uint32_t[2];
  
  // 定义 C 矩阵使用的寄存器类型和数量
  // 8个float寄存器用于存储8x8矩阵的输入和累加结果
  using CRegisters = float[8];

  // FMA操作的静态函数实现
  // 通过内联汇编调用实际的PTX指令
  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      float      & d4, float      & d5, float      & d6, float      & d7,
      uint32_t const& a0, uint32_t const& a1,
      uint32_t const& b0, uint32_t const& b1,
      float  const& c0, float  const& c1, float  const& c2, float  const& c3,
      float  const& c4, float  const& c5, float  const& c6, float  const& c7)
  {
    // 实际的 PTX 指令调用
    // 执行 8x8x4 的矩阵乘法累加操作
    // .m8n8k4 表示操作的尺寸：M=8, N=8, K=4
    // .row.col 表示 A 矩阵是行主序，B 矩阵是列主序
    // .f32.f16.f16.f32 表示数据类型：D(F32), A(F16), B(F16), C(F32)
    asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3, %4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11},"
      "{%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), 
        "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
      :  "r"(a0),  "r"(a1),
         "r"(b0),  "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3), 
        "f"(c4), "f"(c5), "f"(c6), "f"(c7));
  }
};
```

### 2. MMA_Traits 特性结构体

MMA_Traits 为每个 Operation 提供元信息，包括：

- ValTypeD, ValTypeA, ValTypeB, ValTypeC: 逻辑数据类型
- Shape_MNK: MMA 操作的逻辑形状(MxNxK)
- ThrID: MMA 操作中的线程映射
- ALayout, BLayout, CLayout: 线程和值到坐标空间的映射布局

设计原因：

- 分离硬件指令实现和逻辑信息描述
- 提供编译时元信息，支持模板特化
- 描述线程和数据的布局关系，便于内存访问优化

```cpp
template <>
struct MMA_Traits<SM70_8x8x4_F32F16F16F32_NT>
{
  // 定义逻辑数据类型
  using ValTypeD = float;   // 输出矩阵 D 的类型
  using ValTypeA = half_t;  // 输入矩阵 A 的类型
  using ValTypeB = half_t;  // 输入矩阵 B 的类型
  using ValTypeC = float;   // 输入/输出矩阵 C 的类型

  // 定义 MMA 操作的逻辑形状：M=8, N=8, K=4
  using Shape_MNK = Shape<_8,_8,_4>;
  
  // 定义线程ID映射布局
  // 4x2布局，步幅为1和16
  // 表示8个线程组成的quadpair结构
  using ThrID = Layout<Shape <_4, _2>, Stride<_1,_16>>;
  
  // 定义 A 矩阵的线程-值布局
  // Shape <Shape <_4,_2>,_4> 表示 4x2 的线程布局和 4 个值
  // Stride<Stride<_8,_4>,_1> 定义了内存访问的步幅模式
  using ALayout = Layout<Shape <Shape <_4,_2>,_4>, Stride<Stride<_8,_4>,_1>>;
  
  // 定义 B 矩阵的线程-值布局
  // 与 A 矩阵相同，因为它们有相似的访问模式
  using BLayout = Layout<Shape <Shape <_4,_2>,_4>, Stride<Stride<_8,_4>,_1>>;
  
  // 定义 C 矩阵的线程-值布局
  // 更复杂的三维布局，精确描述了8个线程如何访问8个值
  using CLayout = Layout<Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>,
                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>;
};
```

### 3. MMA_Atom 原子操作

MMA_Atom 将 Operation 和 MMA_Traits 结合起来，提供统一接口：

设计原因：

- 统一封装Operation和Traits，提供一致的API
- 支持模板特化，可以针对不同操作进行优化
- 提供make_fragment方法，便于创建适合的张量片段
- 实现call接口，简化MMA操作的调用

```cpp
// 主模板，通过MMA_Traits特化来实现具体功能
template <class MMAOperation>
struct MMA_Atom<MMAOperation> : MMA_Atom<MMA_Traits<MMAOperation>>
{};

// 特化版本，实现具体功能
template <class MMAOperation, class... Args>
struct MMA_Atom<MMA_Traits<MMAOperation, Args...>>
  : MMA_Traits<MMAOperation, Args...>
{
  // 从Traits继承类型定义
  using ValTypeD = typename Traits::ValTypeD;
  using ValTypeA = typename Traits::ValTypeA;
  using ValTypeB = typename Traits::ValTypeB;
  using ValTypeC = typename Traits::ValTypeC;

  using Shape_MNK  = typename Traits::Shape_MNK;
  using ThrID      = typename Traits::ThrID;
  using LayoutC_TV = typename Traits::CLayout;
  using LayoutA_TV = typename Traits::ALayout;
  using LayoutB_TV = typename Traits::BLayout;

  // 主要的调用接口
  // 接受四个张量参数：D(输出), A(输入), B(输入), C(输入/输出)
  template <class TD, class DLayout,
            class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TD, DLayout>      & D,
       Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout> const& C) const
  {
    // 静态断言确保张量是一维的（寄存器级操作）
    static_assert(DLayout::rank == 1, "Expected rank-1 D tensor");
    static_assert(ALayout::rank == 1, "Expected rank-1 A tensor");
    static_assert(BLayout::rank == 1, "Expected rank-1 B tensor");
    static_assert(CLayout::rank == 1, "Expected rank-1 C tensor");

    // 调用底层的mma_unpack函数执行实际操作
    return mma_unpack(static_cast<Traits const&>(*this), D, A, B, C);
  }

  // 三个参数的重载版本，复用C作为输出
  template <class TA, class ALayout,
            class TB, class BLayout,
            class TC, class CLayout>
  CUTE_HOST_DEVICE constexpr
  void
  call(Tensor<TA, ALayout> const& A,
       Tensor<TB, BLayout> const& B,
       Tensor<TC, CLayout>      & C) const
  {
    // 调用四参数版本，将C同时作为输入和输出
    return call(C, A, B, C);
  }
};
```

### 4. TiledMMA 平铺操作

TiledMMA 允许将多个 MMA_Atom 组合成更大的操作，支持多线程协作：

设计原因：

- 支持更大规模的矩阵运算，超越单个MMA指令的能力
- 实现线程级别的并行化，提高硬件利用率
- 提供灵活的平铺策略，适应不同的内存布局需求
- 通过组合多个Atom，构建更复杂的计算模式

```cpp
// TiledMMA模板定义
// MMA_Atom: 基础的MMA操作原子
// AtomLayoutMNK: 在MNK维度上Atom的布局
// PermutationMNK: 应用于每个MNK模式的排列
template <class MMA_Atom,
          class AtomLayoutMNK,
          class PermutationMNK = Tile<Underscore,Underscore,Underscore>>
struct TiledMMA : MMA_Atom
{
  // 从MMA_Atom继承相关类型
  using Atom           = MMA_Atom;
  using AtomShape_MNK  = typename MMA_Atom::Shape_MNK;
  using AtomThrID      = typename MMA_Atom::ThrID;
  using AtomLayoutC_TV = typename MMA_Atom::LayoutC_TV;
  using AtomLayoutA_TV = typename MMA_Atom::LayoutA_TV;
  using AtomLayoutB_TV = typename MMA_Atom::LayoutB_TV;

  // 线程布局，通过将AtomThrID与AtomLayoutMNK进行tiled_product得到
  using ThrLayoutVMNK = decltype(tiled_product(AtomThrID{}, AtomLayoutMNK{}));
  ThrLayoutVMNK thr_layout_vmnk_;

  // 构造函数
  CUTE_HOST_DEVICE constexpr
  TiledMMA(MMA_Atom const& mma_atom = {}, AtomLayoutMNK const& thr_layout_mnk = {})
    : MMA_Atom(mma_atom),
      thr_layout_vmnk_(tiled_product(AtomThrID{}, thr_layout_mnk)) {}

  // 获取线程布局
  CUTE_HOST_DEVICE constexpr auto
  get_thr_layout_vmnk() const {
    return thr_layout_vmnk_;
  }

  // 根据线程索引获取切片
  template <class ThrIdx>
  CUTE_HOST_DEVICE constexpr
  auto
  get_slice(ThrIdx const& thr_idx) const
  {
    // 将线程索引转换为VMNK坐标
    auto thr_vmnk = thr_layout_vmnk_.get_flat_coord(thr_idx);
    // 返回ThrMMA对象，用于特定线程的操作
    return ThrMMA<TiledMMA, decltype(thr_vmnk)>{*this, thr_vmnk};
  }
};

// ThrMMA为特定线程提供操作接口
template <class TiledMMA, class ThrVMNK>
struct ThrMMA : TiledMMA
{
  ThrVMNK thr_vmnk_;

  // 分区C矩阵，为当前线程准备数据
  template <class CTensor>
  CUTE_HOST_DEVICE constexpr
  auto
  partition_C(CTensor&& ctensor) const
  {
    // 创建张量并应用thrfrg_C布局变换
    auto thr_tensor = make_tensor(static_cast<CTensor&&>(ctensor).data(), 
                                  this->thrfrg_C(ctensor.layout()));

    // 构造线程坐标并获取当前线程的数据切片
    auto thr_vmn = make_coord(get<0>(thr_vmnk_), 
                              make_coord(get<1>(thr_vmnk_), get<2>(thr_vmnk_)));
    return thr_tensor(thr_vmn, make_coord(_, repeat<rank<1,1>(thr_tensor)>(_)));
  }
  
  // 类似地实现 partition_A 和 partition_B
};
```

## Algorithm GEMM 接口

CuTe 提供了统一的 GEMM 接口，可以处理不同层次的矩阵乘法操作：

设计原因：

- 提供统一的接口，隐藏底层实现复杂性
- 支持多种数据布局和内存类型
- 根据张量维度和内存类型自动分发到合适的实现
- 实现寄存器级和共享内存级操作的统一调用

```cpp
// 基本 GEMM 接口，使用通用FMA操作
template <class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm(Tensor<TD, DLayout>      & D,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout> const& C)
{
  // 根据张量的数据类型创建合适的MMA操作
  using MMA = MMA_Atom<UniversalFMA<typename Tensor<TD,DLayout>::value_type,
                                    typename Tensor<TA,ALayout>::value_type,
                                    typename Tensor<TB,BLayout>::value_type,
                                    typename Tensor<TC,CLayout>::value_type>>;

  // 调用具体的GEMM实现
  return gemm(MMA{}, D, A, B, C);
}

// 使用特定 MMA_Atom 的 GEMM 接口
template <class MMA,
          class TD, class DLayout,
          class TA, class ALayout,
          class TB, class BLayout,
          class TC, class CLayout>
CUTE_HOST_DEVICE
void
gemm(MMA_Atom<MMA>       const& mma,
     Tensor<TD, DLayout>      & D,
     Tensor<TA, ALayout> const& A,
     Tensor<TB, BLayout> const& B,
     Tensor<TC, CLayout> const& C)
{
  // 根据张量的维度和内存类型进行分发
  // Dispatch [1]: (V) x (V) => (V) - 元素级乘法，直接调用MMA原子操作
  // Dispatch [2]: (M) x (N) => (M,N) - 外积，转换为矩阵乘法
  // Dispatch [3]: (M,K) x (N,K) => (M,N) - 矩阵乘法，添加向量维度
  // Dispatch [4]: (V,M) x (V,N) => (V,M,N) - 批量外积，按元素进行寄存器优化
  // Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N) - 批量矩阵乘法，按K维度循环处理
  
  // 这种分发机制使得GEMM接口可以处理从简单元素操作到复杂批量矩阵运算的各种情况
}
```

## MMA_Atom

MMA_Atom 是 CuTe 中 MMA 操作的基本构建块。它封装了：

- 实际的 MMA 指令（如 HMMA、WMMA 等）
- 输入和输出的布局信息
- MMA 操作的约束条件

MMA_Atom 可以针对不同的硬件架构和数据类型进行优化。

## MMA 操作示例

一个典型的 MMA 操作可能如下所示：

```cpp
// 定义输入张量 A、B 和累加张量 C
auto A_tensor = make_tensor(A_ptr, A_layout);
auto B_tensor = make_tensor(B_ptr, B_layout);
auto C_tensor = make_tensor(C_ptr, C_layout);

// 创建 MMA 操作对象
auto mma_atom = MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>{};

// 获取 MMA 操作的参与者
auto mma_thr = mma_atom.get_thread_slice(thread_idx);

// 创建累加片段
auto accum_fragment = make_fragment_like(C_tensor);
clear(accum_fragment);

// 执行 MMA 操作
mma_thr.call(A_tensor, B_tensor, accum_fragment);
```

## mma_thr.call() 与 cute::gemm() 的区别

虽然可以直接使用 `mma_thr.call()` 执行 MMA 操作，但 CuTe 仍然提供了 `cute::gemm()` 接口，这是因为两者有不同的使用场景和抽象层次：

### mma_thr.call() 的特点：

- **低级接口**：直接操作寄存器级别的张量片段
- **精确控制**：需要手动管理张量的分区和布局
- **硬件相关**：需要明确指定使用的 MMA 操作类型
- **适合场景**：需要精细控制计算过程的高性能场景

### cute::gemm() 的特点：

- **高级接口**：提供统一的 GEMM 接口，自动处理底层细节
- **自动分发**：根据张量的维度和内存类型自动选择合适的实现
- **类型推导**：可以根据输入张量的类型自动推导合适的 MMA 操作
- **灵活适配**：可以处理从简单元素操作到复杂批量矩阵运算的各种情况
- **适合场景**：通用的矩阵乘法计算，简化开发流程

### 使用建议：

```cpp
// 当需要精细控制时，使用 mma_thr.call()
auto mma_atom = MMA_Atom<SM70_8x8x4_F32F16F16F32_NT>{};
auto mma_thr = mma_atom.get_thread_slice(thread_idx);
mma_thr.call(A_frag, B_frag, C_frag);

// 当需要通用接口时，使用 cute::gemm()
cute::gemm(D_tensor, A_tensor, B_tensor, C_tensor);
```

总的来说，`mma_thr.call()` 提供了更底层、更精确的控制，而 `cute::gemm()` 提供了更高级、更通用的接口。开发者可以根据具体需求选择合适的接口。

## GMMA Descriptor 与 Swizzle 信息

对于 Hopper (SM90) 架构中的 GMMA 操作，请参考专门的文档：[CuTe WGmma SM90](cute_wgmma_sm90.md)

## 累加片段 (Accumulator Fragment)

在 MMA 操作中，累加片段是非常重要的概念。它代表了累加器寄存器中的数据块。CuTe 提供了专门的类型和操作来处理累加片段：

```cpp
// 创建累加片段
auto accum_fragment = make_fragment_like(C_tensor);

// 初始化累加片段
clear(accum_fragment);

// 执行多次 MMA 操作累加结果
mma_thr.call(A_tensor, B_tensor, accum_fragment, accum_fragment);
```

## 与 Copy 操作的协同

在实际应用中，MMA 操作通常与 Copy 操作协同工作：

- 使用 Copy 操作将数据从全局内存加载到共享内存或寄存器
- 使用 MMA 操作执行计算
- 使用 Copy 操作将结果从寄存器写回到全局内存

这种协同工作模式充分利用了 GPU 的内存层次结构和计算能力。

## 不同架构的支持

CuTe 支持多种 NVIDIA GPU 架构的 MMA 指令：

### Volta (SM70)

- 使用 HMMA 指令
- 8个线程的 quadpair 协作完成 8x8x4 的矩阵乘法

### Turing (SM75)

- 增强的 HMMA 指令支持

### Ampere (SM80)

- 更多的 MMA 指令变体
- 对稀疏矩阵乘法的支持

### Hopper (SM90)

- 引入了新一代的 GMMA (Group MMA) 指令
- 支持更大的矩阵操作 (如 64x128x16)
- warpgroup 级别的协作 (128个线程)

## 高级特性

CuTe 的 MMA 操作具有以下高级特性：

### 可扩展性

通过 TiledMMA，可以轻松地扩展基本的 MMA 操作以适应更大的矩阵计算需求。

### 灵活的布局支持

CuTe 的 Layout 系统使得可以灵活地处理不同的数据布局，包括行主序、列主序以及自定义布局。

### 类型安全

通过模板和类型系统，确保在编译时就能发现类型不匹配的错误。

这些抽象使得开发者可以编写高效且可维护的 CUDA 代码，同时充分利用现代 GPU 的计算能力。