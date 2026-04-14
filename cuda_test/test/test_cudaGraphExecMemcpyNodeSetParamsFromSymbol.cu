#include <cuda_runtime.h>
#include <UPTK_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

#define CHECK(call)                                                   \
    do {                                                              \
        UPTKError_t err = call;                                       \
        if (err != UPTKSuccess) {                                     \
            std::cerr << "CUDA error: " << UPTKGetErrorString(err)    \
                      << " at " << __FILE__ << ":" << __LINE__        \
                      << std::endl;                                   \
            exit(1);                                                  \
        }                                                             \
    } while (0)

// 设备符号
__device__ int d_symbol[4];

int main() {
    const int N = 4;
    int h_src[N] = {1, 2, 3, 4};
    int h_dst[N] = {0};

    int *d_dst;
    CHECK(UPTKMalloc(&d_dst, N * sizeof(int)));

    // 初始化 symbol
    CHECK(UPTKMemcpyToSymbol(d_symbol, h_src, N * sizeof(int)));

    // 创建 graph
    UPTKGraph_t graph;
    CHECK(UPTKGraphCreate(&graph, 0));

    // 创建 memcpy node（从 symbol 拷贝到 device memory）
    UPTKGraphNode_t memcpyNode;

    UPTKMemcpy3DParms copyParams = {0};
    copyParams.srcArray = NULL;
    copyParams.srcPos = make_UPTKPos(0, 0, 0);
    copyParams.srcPtr = make_UPTKPitchedPtr(NULL, 0, 0, 0); // 占位
    copyParams.dstPtr = make_UPTKPitchedPtr(d_dst, N * sizeof(int), N, 1);
    copyParams.extent = make_UPTKExtent(N * sizeof(int), 1, 1);
    copyParams.kind = UPTKMemcpyDeviceToDevice;

    // 注意：这里用 FromSymbol API 创建节点
    CHECK(UPTKGraphAddMemcpyNodeFromSymbol(
        &memcpyNode,
        graph,
        NULL,
        0,
        d_dst,
        d_symbol,
        N * sizeof(int),
        0,
        UPTKMemcpyDeviceToDevice));

    // 实例化 graph
    UPTKGraphExec_t graphExec;
    CHECK(UPTKGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // ====== 修改 symbol 数据 ======
    int h_src_new[N] = {10, 20, 30, 40};
    CHECK(UPTKMemcpyToSymbol(d_symbol, h_src_new, N * sizeof(int)));

    // ====== 修改 graphExec 中 memcpy 参数 ======
    int *d_dst_new;
    CHECK(UPTKMalloc(&d_dst_new, N * sizeof(int)));

    CHECK(UPTKGraphExecMemcpyNodeSetParamsFromSymbol(
        graphExec,
        memcpyNode,
        d_dst_new,     // 新目标地址
        d_symbol,
        N * sizeof(int),
        0,
        UPTKMemcpyDeviceToDevice));

    // 执行 graph
    UPTKStream_t stream;
    CHECK(UPTKStreamCreate(&stream));
    CHECK(UPTKGraphLaunch(graphExec, stream));
    CHECK(UPTKStreamSynchronize(stream));

    // 拷回 host 验证
    CHECK(UPTKMemcpy(h_dst, d_dst_new, N * sizeof(int), UPTKMemcpyDeviceToHost));

    // 验证结果
    for (int i = 0; i < N; i++) {
        assert(h_dst[i] == h_src_new[i]);
    }

    //std::cout << "Test PASSED!" << std::endl;
    printf("test_cudaGraphExecMemcpyNodeSetParamsFromSymbol PASS\n");

    // 清理
    UPTKFree(d_dst);
    UPTKFree(d_dst_new);
    UPTKGraphExecDestroy(graphExec);
    UPTKGraphDestroy(graph);
    UPTKStreamDestroy(stream);

    return 0;
}
