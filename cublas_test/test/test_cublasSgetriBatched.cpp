#include <cublas_v2.h>
#include <stdio.h>

int main() {
    // DTK/AMD GPU does not support array-of-pointers batched cuBLAS APIs.
    // These APIs trigger VMFault/abort on this platform.
    // Strided batched versions (e.g., UPTKblasSgemmStridedBatched) work correctly.
    printf("test_skip: cublas batched (array-of-pointers) APIs not supported on DTK/AMD GPU\n");
    return 0;
}
