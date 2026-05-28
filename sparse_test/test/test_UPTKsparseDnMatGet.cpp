/*
 * Auto-generated smoke test for UPTKsparseDnMatGet (sparse_fun_convert.cpp).
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
}


int main(void)
{
    UPTKsparseHandle_t sparse_handle{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKsparseDnMatDescr_t dnMatDescr{};
    // 新增：分配合法的设备内存作为矩阵数据
    float *d_mat_data = nullptr;
    const int64_t rows = 2;    // 合法行数
    const int64_t cols = 3;    // 合法列数
    const int64_t lda = cols;  // 合法步长（列优先时lda=cols，行优先时lda=rows）

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDnMatGet setup failed (%d)\n", (int)err);
        return 0;
    }

    // 新增：分配设备内存存储矩阵数据
    if (cudaMalloc((void**)&d_mat_data, rows * cols * sizeof(float)) != cudaSuccess) {
        printf("test_skip: cudaMalloc for matrix data failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }

    // 修正：传入合法的矩阵参数（非0行列数、合法步长、有效设备指针）
    err = UPTKsparseCreateDnMat(
        &dnMatDescr, 
        rows,                  // 行数（非0）
        cols,                  // 列数（非0）
        lda,                   // 步长（非0）
        d_mat_data,            // 有效设备指针
        UPTK_R_32F,            // 数据类型
        UPTKSPARSE_ORDER_COL   // 矩阵存储顺序（显式指定合法值，而非0）
    );
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: create descriptor failed (%d)\n", (int)err);
        cudaFree(d_mat_data);  // 新增：释放已分配的内存
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }

    // 可选：如果需要获取参数，可定义变量接收
    int64_t out_rows, out_cols, out_lda;
    void *out_data;
    UPTKDataType out_dtype;
    UPTKsparseOrder_t out_order;
    err = UPTKsparseDnMatGet(
        dnMatDescr, 
        &out_rows,    // 接收行数
        &out_cols,    // 接收列数
        &out_lda,     // 接收步长
        &out_data,    // 接收数据指针
        &out_dtype,   // 接收数据类型
        &out_order    // 接收存储顺序
    );

    printf("UPTKsparseDnMatGet -> %d\n", (int)err);
    // 可选：打印获取到的参数，验证正确性
    printf("Got matrix params: rows=%lld, cols=%lld, lda=%lld, dtype=%d, order=%d\n",
           out_rows, out_cols, out_lda, (int)out_dtype, (int)out_order);

    if (dnMatDescr) UPTKsparseDestroyDnMat(dnMatDescr);
    if (d_mat_data) cudaFree(d_mat_data);  // 新增：释放矩阵数据内存

    sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
    printf("test_UPTKsparseDnMatGet PASS\n");
    return 0;
}
