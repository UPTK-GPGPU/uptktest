/*
 * Auto-generated smoke test for UPTKrtcVersion (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_rtc.h>
#include <stdio.h>

int main(void)
{
    int maj = 0, min = 0;
    UPTKrtcResult err = UPTKrtcVersion(&maj, &min);
    printf("UPTKrtcVersion -> %d (%d.%d)\n", (int)err, maj, min);
    printf("test_UPTKrtcVersion PASS\n");
    return 0;
}
