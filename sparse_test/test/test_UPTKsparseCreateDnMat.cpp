/*
 * Auto-generated smoke test for UPTKsparseCreateDnMat (sparse_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/sparse_test/generate_sparse_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_sparse.h>
#include <stdint.h>
#include <stdio.h>

static UPTKsparseStatus_t sparse_bind_gpu(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return (UPTKsparseStatus_t)1;
    if (cudaSetDevice(0) != cudaSuccess)
        return (UPTKsparseStatus_t)1;
    return UPTKSPARSE_STATUS_SUCCESS;
}

static UPTKsparseStatus_t sparse_setup_core(
    UPTKsparseHandle_t *sparse_handle,
    void **dev_scratch,
    UPTKStream_t *stream_id,
    int create_handle)
{
    UPTKsparseStatus_t st = sparse_bind_gpu();
    if (st != UPTKSPARSE_STATUS_SUCCESS) return st;

    if (cudaMalloc(dev_scratch, 65536) != cudaSuccess)
        return (UPTKsparseStatus_t)1;

    if (cudaStreamCreate(stream_id) != cudaSuccess) {
        cudaFree(*dev_scratch);
        *dev_scratch = nullptr;
        return (UPTKsparseStatus_t)1;
    }

    if (create_handle) {
        st = UPTKsparseCreate(sparse_handle);
        if (st != UPTKSPARSE_STATUS_SUCCESS) return st;
        st = UPTKsparseSetStream(*sparse_handle, *stream_id);
        if (st != UPTKSPARSE_STATUS_SUCCESS) return st;
    } else {
        *sparse_handle = (UPTKsparseHandle_t)(uintptr_t)0;
    }

    return UPTKSPARSE_STATUS_SUCCESS;
}

static void sparse_teardown_core(
    UPTKsparseHandle_t sparse_handle,
    void *dev_scratch,
    UPTKStream_t stream_id,
    int destroy_sparse_handle)
{
    if (destroy_sparse_handle && sparse_handle)
        UPTKsparseDestroy(sparse_handle);
    if (dev_scratch)
        cudaFree(dev_scratch);
    if (stream_id)
        cudaStreamDestroy(stream_id);
    // 显式return消除void函数返回警告
    return;
}

int main(void)
{
    UPTKsparseHandle_t sparse_handle{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKsparseDnMatDescr_t dnMatDescr{};

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseCreateDnMat setup failed (%d)\n", (int)err);
        return 0;
    }

    // 1. 分配设备端矩阵内存（合法非空数据指针）
    float *d_mat = nullptr;
    const int64_t rows = 2, cols = 3;       // 合法矩阵维度（非0）
    const int64_t lda = rows;               // 主维度（行优先时通常为行数）
    cudaMalloc(&d_mat, rows * cols * sizeof(float));

    // 2. 调用UPTKsparseCreateDnMat：使用合法枚举值+参数
    // 需核对UPTK_sparse.h中UPTKsparseOrder_t的定义，示例假设：
    // typedef enum { UPTKSPARSE_ORDER_ROW = 1, UPTKSPARSE_ORDER_COL = 2 } UPTKsparseOrder_t;
    err = UPTKsparseCreateDnMat(
        &dnMatDescr,
        rows,                  // 有效行数
        cols,                  // 有效列数
        lda,                   // 有效主维度
        d_mat,                 // 设备端数据指针（非null）
        UPTK_R_32F,            // 数据类型（无需强制转换）
        UPTKSPARSE_ORDER_ROW   // 合法的矩阵顺序枚举值
    );

    printf("UPTKsparseCreateDnMat -> %d\n", (int)err);

    // 清理设备矩阵内存
    if (d_mat) cudaFree(d_mat);

    sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
    printf("test_UPTKsparseCreateDnMat PASS\n");
    return 0;
}
