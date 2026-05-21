/*
 * Auto-generated smoke test for UPTKrtcGetSupportedArchs (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_rtc.h>
#include <stdio.h>

int main(void)
{
    int arches[128];
    UPTKrtcResult err = UPTKrtcGetSupportedArchs(arches);
    printf("UPTKrtcGetSupportedArchs -> %d\n", (int)err);
    printf("test_UPTKrtcGetSupportedArchs PASS\n");
    return 0;
}
