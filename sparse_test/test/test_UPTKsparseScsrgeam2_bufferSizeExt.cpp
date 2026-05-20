/*
 * Auto-generated smoke test for UPTKsparseScsrgeam2_bufferSizeExt (sparse_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/sparse_test/generate_sparse_tests.ps1
 */
#include <UPTK_sparse.h>
#include <cuda_runtime.h>
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
    UPTKsparseMatDescr_t descrB{};
    UPTKsparseMatDescr_t descrC{};

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt setup failed (%d)\n", (int)err);
        return 0;
    }


    err = UPTKsparseCreateMatDescr(&descrA);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt UPTKsparseCreateMatDescr(descrA) failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
        return 0;
    }
    err = UPTKsparseSetMatIndexBase(descrA, UPTKSPARSE_INDEX_BASE_ZERO);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt SetMatIndexBase(descrA)\n");
        UPTKsparseDestroyMatDescr(descrA); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }
    err = UPTKsparseSetMatType(descrA, UPTKSPARSE_MATRIX_TYPE_GENERAL);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt SetMatType(descrA)\n");
        UPTKsparseDestroyMatDescr(descrA); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }

    err = UPTKsparseCreateMatDescr(&descrB);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt UPTKsparseCreateMatDescr(descrB) failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
        return 0;
    }
    err = UPTKsparseSetMatIndexBase(descrB, UPTKSPARSE_INDEX_BASE_ZERO);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt SetMatIndexBase(descrB)\n");
        UPTKsparseDestroyMatDescr(descrB); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }
    err = UPTKsparseSetMatType(descrB, UPTKSPARSE_MATRIX_TYPE_GENERAL);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt SetMatType(descrB)\n");
        UPTKsparseDestroyMatDescr(descrB); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }

    err = UPTKsparseCreateMatDescr(&descrC);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt UPTKsparseCreateMatDescr(descrC) failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
        return 0;
    }
    err = UPTKsparseSetMatIndexBase(descrC, UPTKSPARSE_INDEX_BASE_ZERO);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt SetMatIndexBase(descrC)\n");
        UPTKsparseDestroyMatDescr(descrC); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }
    err = UPTKsparseSetMatType(descrC, UPTKSPARSE_MATRIX_TYPE_GENERAL);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseScsrgeam2_bufferSizeExt SetMatType(descrC)\n");
        UPTKsparseDestroyMatDescr(descrC); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }

    err = UPTKsparseScsrgeam2_bufferSizeExt(sparse_handle, 0, 0, (const float*)dev_scratch, descrA, 0, (const float*)dev_scratch, (const int*)dev_scratch, (const int*)dev_scratch, (const float*)dev_scratch, descrB, 0, (const float*)dev_scratch, (const int*)dev_scratch, (const int*)dev_scratch, descrC, (const float*)dev_scratch, (const int*)dev_scratch, (const int*)dev_scratch, (size_t*)dev_scratch);

    printf("UPTKsparseScsrgeam2_bufferSizeExt -> %d\n", (int)err);



    if (descrA) UPTKsparseDestroyMatDescr(descrA);

    if (descrB) UPTKsparseDestroyMatDescr(descrB);

    if (descrC) UPTKsparseDestroyMatDescr(descrC);
    sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
    printf("test_UPTKsparseScsrgeam2_bufferSizeExt PASS\n");
    return 0;
}
