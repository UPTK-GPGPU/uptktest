/*
 * Auto-generated smoke test for UPTKrtcGetErrorString (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <UPTK_rtc.h>
#include <stdio.h>

int main(void)
{
    const char *s = UPTKrtcGetErrorString(UPTKRTC_SUCCESS);
    printf("UPTKrtcGetErrorString -> %s\n", s ? s : "(null)");
    printf("test_UPTKrtcGetErrorString PASS\n");
    return 0;
}
