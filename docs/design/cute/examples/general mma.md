
一个 mapping 就够了
g2s，dmu slice 会支持

## workload
mnk: 128, 128, 64
tiled_mnk: 64, 32, 32

A: (128, 64):(64, 1)
thr_a: ((_64,_32),2,2):((64,_1),4096,_32)

## 如何封装到 cute api

传递坐标是为了生成新的 tensor 


## TODO
- 尽快实现第一版 cuda custom copy 
- 迁移到 T 芯片
- 阅读 tiled mma 里的代码逻辑，整理文档
- layout 计算器，做一个项目

