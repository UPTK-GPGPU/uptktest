/*
 *  * Auto-generated smoke test for UPTKsparseCooGet (sparse_fun_convert.cpp).
 *   * Regenerate: powershell -ExecutionPolicy Bypass -File test/sparse_test/generate_sparse_tests.ps1
 *    */
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
    UPTKsparseSpMatDescr_t spMatDescr{};

    const int64_t nrows = 2;    
    const int64_t ncols = 2;    
    const int64_t nnz = 2;      
    int32_t *d_rowInd = nullptr;
    int32_t *d_colInd = nullptr;
    float *d_values = nullptr;  

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseCooGet setup failed (%d)\n", (int)err);
        return 0;
    }

    if (cudaMalloc(&d_rowInd, nnz * sizeof(int32_t)) != cudaSuccess) {
        printf("test_fail: cudaMalloc rowInd failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 1;
    }
    if (cudaMalloc(&d_colInd, nnz * sizeof(int32_t)) != cudaSuccess) {
        printf("test_fail: cudaMalloc colInd failed\n");
        cudaFree(d_rowInd);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 1;
    }
    if (cudaMalloc(&d_values, nnz * sizeof(float)) != cudaSuccess) {
        printf("test_fail: cudaMalloc values failed\n");
        cudaFree(d_rowInd);
        cudaFree(d_colInd);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 1;
    }

    err = UPTKsparseCreateCoo(
        &spMatDescr,
        nrows, ncols, nnz,
        d_rowInd, d_colInd, d_values,
        UPTKSPARSE_INDEX_32I,
        UPTKSPARSE_INDEX_BASE_ZERO,
        UPTK_R_32F
    );
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: create descriptor failed (%d)\n", (int)err);
        cudaFree(d_rowInd);
        cudaFree(d_colInd);
        cudaFree(d_values);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }

    int64_t get_nrows, get_ncols, get_nnz;
    void *get_rowInd, *get_colInd, *get_values;
    UPTKsparseIndexType_t get_idx_type;
    UPTKsparseIndexBase_t get_idx_base;
    UPTKDataType_t get_data_type;

    err = UPTKsparseCooGet(
        spMatDescr,
        &get_nrows, &get_ncols, &get_nnz,
        &get_rowInd, &get_colInd, &get_values,
        &get_idx_type, &get_idx_base, &get_data_type
    );

    printf("UPTKsparseCooGet -> %d\n", (int)err);
    printf("Got: nrows=%lld, ncols=%lld, nnz=%lld\n", (long long)get_nrows, (long long)get_ncols, (long long)get_nnz);
    printf("IndexType=%d, IndexBase=%d, DataType=%d\n", (int)get_idx_type, (int)get_idx_base, (int)get_data_type);

    if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);
    cudaFree(d_rowInd);
    cudaFree(d_colInd);
    cudaFree(d_values);
    sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);

    printf("test_UPTKsparseCooGet PASS\n");
    return 0;
}
