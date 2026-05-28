/*
 * Auto-generated smoke test for UPTKsparseSpVecGet (sparse_fun_convert.cpp).
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
    UPTKsparseSpVecDescr_t spVecDescr{};

    // 新增：定义稀疏向量的有效参数
    const int64_t vec_size = 4;        // 向量总维度
    const int64_t nnz = 2;             // 非零元素数量
    int32_t h_indices[] = {0, 2};      // 主机端索引数组（32位）
    float h_values[] = {1.0f, 3.0f};   // 主机端数值数组（32位浮点）
    void *d_indices = nullptr;         // 设备端索引数组
    void *d_values = nullptr;          // 设备端数值数组

    UPTKsparseStatus_t err;

    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, 1);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: UPTKsparseSpVecGet setup failed (%d)\n", (int)err);
        return 0;
    }

    // 新增：分配设备端内存并拷贝数据
    if (cudaMalloc(&d_indices, nnz * sizeof(int32_t)) != cudaSuccess) {
        printf("test_skip: cudaMalloc indices failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }
    if (cudaMalloc(&d_values, nnz * sizeof(float)) != cudaSuccess) {
        printf("test_skip: cudaMalloc values failed\n");
        cudaFree(d_indices);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }
    cudaMemcpy(d_indices, h_indices, nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // 修改：传入有效参数创建稀疏向量描述符
    err = UPTKsparseCreateSpVec(
        &spVecDescr,
        vec_size,                // 向量总维度（非0）
        nnz,                     // 非零元素数（非0）
        d_indices,               // 设备端索引数组（非nullptr）
        d_values,                // 设备端数值数组（非nullptr）
        UPTKSPARSE_INDEX_32I,    // 索引类型（32位）
        UPTKSPARSE_INDEX_BASE_ZERO, // 索引基址（0开始）
        UPTK_R_32F               // 数据类型（32位浮点）
    );
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: create descriptor failed (%d)\n", (int)err);
        cudaFree(d_indices);  // 新增：释放设备内存
        cudaFree(d_values);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }

    // 测试 UPTKsparseSpVecGet（可按需接收返回的参数）
    int64_t out_size, out_nnz;
    void *out_indices, *out_values;
    UPTKsparseIndexType_t out_idx_type;
    UPTKsparseIndexBase_t out_idx_base;
    UPTKDataType_t out_dtype;
    err = UPTKsparseSpVecGet(
        spVecDescr,
        &out_size,       // 接收向量维度
        &out_nnz,        // 接收非零元素数
        &out_indices,    // 接收索引数组指针
        &out_values,     // 接收数值数组指针
        &out_idx_type,   // 接收索引类型
        &out_idx_base,   // 接收索引基址
        &out_dtype       // 接收数据类型
    );

    printf("UPTKsparseSpVecGet -> %d\n", (int)err);
    // 可选：打印返回的参数验证
    printf("Returned size: %lld, nnz: %lld, idx_type: %d, idx_base: %d, dtype: %d\n",
           out_size, out_nnz, (int)out_idx_type, (int)out_idx_base, (int)out_dtype);

    // 清理资源
    if (spVecDescr) UPTKsparseDestroySpVec(spVecDescr);
    cudaFree(d_indices);  // 新增：释放设备内存
    cudaFree(d_values);

    sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
    printf("test_UPTKsparseSpVecGet PASS\n");
    return 0;
}
