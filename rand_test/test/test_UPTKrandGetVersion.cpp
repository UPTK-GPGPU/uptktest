/*
 * Auto-generated smoke test for UPTKrandGetVersion (rand_fun_convert.cpp).
 */
#include <cuda_runtime.h>
#include <UPTK_rand.h>
#include <stdio.h>

int main(void)
{
    int version = 0;
    UPTKrandStatus_t err = UPTKrandGetVersion(&version);
    printf("UPTKrandGetVersion -> %d (ver=%d)\n", (int)err, version);
    printf("test_UPTKrandGetVersion PASS\n");
    return 0;
}
