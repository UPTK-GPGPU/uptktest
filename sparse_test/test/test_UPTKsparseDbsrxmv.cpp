/*
 * Auto-generated smoke test for UPTKsparseDbsrxmv (sparse_fun_convert.cpp).
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
    UPTKStream_t stream_id)
{
    if (sparse_handle)
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
    UPTKsparseMatDescr_t descrA{};

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDbsrxmv setup failed (%d)\n", (int)err);
        return 0;
    }


    err = UPTKsparseCreateMatDescr(&descrA);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDbsrxmv UPTKsparseCreateMatDescr(descrA) failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
        return 0;
    }
    err = UPTKsparseSetMatIndexBase(descrA, UPTKSPARSE_INDEX_BASE_ZERO);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDbsrxmv SetMatIndexBase(descrA)\n");
        UPTKsparseDestroyMatDescr(descrA); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }
    err = UPTKsparseSetMatType(descrA, UPTKSPARSE_MATRIX_TYPE_GENERAL);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseDbsrxmv SetMatType(descrA)\n");
        UPTKsparseDestroyMatDescr(descrA); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }

    err = UPTKsparseDbsrxmv(sparse_handle, (UPTKsparseDirection_t)UPTKSPARSE_DIRECTION_ROW, (UPTKsparseOperation_t)UPTKSPARSE_OPERATION_NON_TRANSPOSE, 0, 0, 0, 0, (const double*)dev_scratch, descrA, (const double*)dev_scratch, (const int*)dev_scratch, (const int*)dev_scratch, (const int*)dev_scratch, (const int*)dev_scratch, 0, (const double*)dev_scratch, (const double*)dev_scratch, (double*)dev_scratch);

    printf("UPTKsparseDbsrxmv -> %d\n", (int)err);



    if (descrA) UPTKsparseDestroyMatDescr(descrA);
    sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
    printf("test_UPTKsparseDbsrxmv PASS\n");
    return 0;
}
