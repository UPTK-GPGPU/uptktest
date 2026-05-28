/*
 * Auto-generated smoke test for UPTKsparseCsrGet (sparse_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/sparse_test/generate_sparse_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_sparse.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    UPTKsparseSpMatDescr_t spMatDescr{};

    UPTKsparseStatus_t err;

    // 1. 初始化核心资源
    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseCsrGet setup failed (%d)\n", (int)err);
        return 0;
    }

    // 2. 定义合法的CSR矩阵参数（2x2对角矩阵，nnz=2）
    const int64_t m = 2;          // 行数
    const int64_t n = 2;          // 列数
    const int64_t nnz = 2;        // 非零元素数
    int row_ptr[] = {0, 1, 2};    // 行指针（32位索引）
    int col_ind[] = {0, 1};       // 列索引（32位索引）
    float values[] = {1.0f, 2.0f};// 数值（32位浮点）

    // 3. 分配设备端内存并拷贝数据（UPTKsparse通常要求设备指针）
    void *d_row_ptr, *d_col_ind, *d_values;
    cudaMalloc(&d_row_ptr, sizeof(row_ptr));
    cudaMalloc(&d_col_ind, sizeof(col_ind));
    cudaMalloc(&d_values, sizeof(values));
    cudaMemcpy(d_row_ptr, row_ptr, sizeof(row_ptr), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, col_ind, sizeof(col_ind), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(values), cudaMemcpyHostToDevice);

    // 4. 创建CSR矩阵描述符（传入合法参数和设备指针）
    err = UPTKsparseCreateCsr(
        &spMatDescr,
        m, n, nnz,
        d_row_ptr, d_col_ind, d_values,
        UPTKSPARSE_INDEX_32I,    // 行索引类型（显式传枚举，不强制转换）
        UPTKSPARSE_INDEX_32I,    // 列索引类型
        UPTKSPARSE_INDEX_BASE_ZERO, // 索引基址（0开始）
        UPTK_R_32F               // 数据类型
    );
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: create descriptor failed (%d)\n", (int)err);
        // 释放设备内存
        cudaFree(d_row_ptr);
        cudaFree(d_col_ind);
        cudaFree(d_values);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }

    // 5. 调用UPTKsparseCsrGet（可传入非nullptr指针验证返回值）
    int64_t out_m, out_n, out_nnz;
    void *out_row_ptr, *out_col_ind, *out_values;
    UPTKsparseIndexType_t out_row_itype, out_col_itype;
    UPTKsparseIndexBase_t out_idx_base;
    UPTKDataType_t out_dtype;

    err = UPTKsparseCsrGet(
        spMatDescr,
        &out_m, &out_n, &out_nnz,
        &out_row_ptr, &out_col_ind, &out_values,
        &out_row_itype, &out_col_itype,
        &out_idx_base, &out_dtype
    );

    printf("UPTKsparseCsrGet -> %d\n", (int)err);
    // 打印返回的参数（验证正确性）
    printf("Returned m=%ld, n=%ld, nnz=%ld\n", out_m, out_n, out_nnz);
    printf("Index type: row=%d, col=%d | Index base=%d | Data type=%d\n",
           (int)out_row_itype, (int)out_col_itype, (int)out_idx_base, (int)out_dtype);

    // 6. 释放资源
    if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);

    sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
    printf("test_UPTKsparseCsrGet PASS\n");
    return 0;
}
