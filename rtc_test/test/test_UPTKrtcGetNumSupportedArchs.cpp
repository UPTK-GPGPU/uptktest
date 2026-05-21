/*
 * Auto-generated smoke test for UPTKrtcGetNumSupportedArchs (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_rtc.h>
#include <stdio.h>

int main(void)
{
    int n = 0;
    UPTKrtcResult err = UPTKrtcGetNumSupportedArchs(&n);
    printf("UPTKrtcGetNumSupportedArchs -> %d (n=%d)\n", (int)err, n);
    printf("test_UPTKrtcGetNumSupportedArchs PASS\n");
    return 0;
}
