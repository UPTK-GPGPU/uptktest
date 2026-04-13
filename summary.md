# CUDA API 测试工作总结

## 项目概述
- **环境**: DTK/AMD GPU (release 11.8, HIP-based CUDA-compatible)
- **头文件目录**: /root/CUDA-test/cuda_header/
- **测试根目录**: /root/CUDA-test/UPTK-CUDA-test/
- **构建系统**: CMake + Make

## 测试文件统计
| 库 | API数量 | 测试文件数 | 文件扩展名 |
|---|---------|-----------|-----------|
| CUDA Runtime | 347 | 344 | .cpp + .cu |
| cuBLAS | 326 | 326 | .cpp |
| cuFFT | 52 | 52 | .cpp |
| NCCL | 19 | 19 | .cpp |
| **总计** | **744** | **741** | |

## 构建结果
- **编译通过率**: 100% (741/741 全部编译成功)
- 修复了约 50+ 个编译错误，包括：
  - 函数签名不匹配（enum 名称、参数数量、参数类型）
  - 缺少头文件（stdlib.h, stdint.h）
  - .cpp 文件需要改为 .cu（含 texture/surface/__device__/__global__）
  - cuBLAS 复杂类型参数错误（alpha/beta 类型混用）
  - cuBLAS Xt API 签名错误

## 测试结果
- **通过**: 656/741 (89%)
- **失败**: 85/741 (11%)

## 失败原因分类

### 1. cuBLAS Batched 操作 (23个) - DTK 不支持
- cublas*SgemmBatched, *DgemmBatched, *CgemmBatched, *ZgemmBatched, *HgemmBatched
- cublas*SgemvBatched, *DgemvBatched, *CgemvBatched, *ZgemvBatched
- cublas*SgetrfBatched, *DgetrfBatched, *CgetrfBatched, *ZgetrfBatched
- cublas*SgetriBatched, *DgetriBatched, *CgetriBatched
- cublas*StrsmBatched, *DtrsmBatched, *CtrsmBatched, *ZtrsmBatched
- cublasCmatinvBatched, cublasZmatinvBatched
- cublasGemmBatchedEx
- cublasSetMathMode

### 2. cuBLAS Xt 操作 (9个) - DTK 不支持
- cublasXtCreate, cublasXtDestroy
- cublasXtGetBlockDim, cublasXtSetBlockDim
- cublasXtGetNumBoards, cublasXtGetPinningMemMode, cublasXtSetPinningMemMode
- cublasXtSetCpuRatio, cublasXtSetCpuRoutine

### 3. CUDA Runtime - 平台差异 (18个)
- **Stream Capture**: cudaStreamBeginCapture, cudaStreamEndCapture, cudaStreamCopyAttributes, cudaStreamGetAttribute, cudaStreamSetAttribute
- **Deprecated Texture**: cudaGetTextureObjectResourceViewDesc, cudaGetTextureObjectTextureDesc, cudaUnbindTexture
- **Device Symbols**: cudaGetSymbolAddress, cudaGetSymbolSize, cudaFuncSetAttribute
- **Other**: cudaGetDriverEntryPoint, cudaPointerGetAttributes, cudaStreamWaitEvent, cudaThreadGetLimit, cudaThreadSetLimit, cudaUserObjectCreate, cudaGraphKernelNodeSetAttribute, cudaGetSurfaceObjectResourceDesc

## 关键经验教训 (learnings.md)
1. DTK batched API 全部 abort，无法通过代码修复
2. DTK Xt API 全部不支持
3. cuBLAS TRMM 需要 3 个矩阵参数 (A, B, C)
4. cuBLAS gelsBatched 的 info 是 host 指针，devInfoArray 是 device 指针
5. cuBLAS getriBatched 没有 workspace 参数
6. cuBLAS tpttr/trttp 没有 info 参数
7. cuBLAS Xt Cherk/Zherk 的 beta 是实数不是复数
8. cuBLAS Xt spmm 的 side 参数在 fill 参数之前
9. cudaCreateChannelDesc 模板只接受 1 个类型参数
10. cudaDeviceGetAttribute 使用 MaxGridDimX/Y/Z 而非 MaxGridSize
11. cudaProfilerStart/Stop 在 DTK 头文件中不可用
12. 含 __device__/__global__/texture/surface 的文件必须用 .cu 扩展名

## 下一步优化方向
1. 分析 batched 失败是否可以改为 test_skip 而非 abort
2. 分析 stream capture 相关失败是否可以绕过
3. 分析 deprecated texture API 是否可以简化测试
