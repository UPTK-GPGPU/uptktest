/*
 * Auto-generated smoke test for UPTKrandGetProperty (rand_fun_convert.cpp).
 */
#include <UPTK_rand.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

int main(void)
{
    int value = 0;
    UPTKrandStatus_t err = UPTKrandGetProperty((libraryPropertyType)MAJOR_VERSION, &value);
    printf("UPTKrandGetProperty -> %d\n", (int)err);
    printf("test_UPTKrandGetProperty PASS\n");
    return 0;
}
