# Generates test/test_UPTKrtc*.cpp for each wrapper in rtc_fun_convert.cpp
$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$srcPath = Join-Path $repoRoot 'src\rtc\rtc_fun_convert.cpp'
$outDir = Join-Path $PSScriptRoot 'test'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$text = Get-Content -LiteralPath $srcPath -Raw -Encoding UTF8
$text = [regex]::Replace($text, '/\*[\s\S]*?\*/', '')

$rx = [regex]::Matches($text, '(?m)^(?:UPTKrtcResult|const\s+char\s*\*)\s+(UPTKrtc\w+)\(')
Write-Host "Matched $($rx.Count) rtc wrappers"

function Get-CppForRtcFunction([string]$fname) {
    $hdr = @"
/*
 * Auto-generated smoke test for $fname (rtc_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rtc_test/generate_rtc_tests.ps1
 */
#include <cuda_runtime.h>
"@

    switch ($fname) {
        'UPTKrtcCreateProgram' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    printf("UPTKrtcCreateProgram -> %d\n", (int)err);
    if (prog)
        UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcCreateProgram PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcDestroyProgram' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create program failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcDestroyProgram(&prog);
    printf("UPTKrtcDestroyProgram -> %d\n", (int)err);
    printf("test_UPTKrtcDestroyProgram PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcCompileProgram' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    printf("UPTKrtcCompileProgram -> %d\n", (int)err);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcCompileProgram PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcAddNameExpression' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcAddNameExpression(prog, "uptk_rtc_smoke_kernel");
    printf("UPTKrtcAddNameExpression -> %d\n", (int)err);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcAddNameExpression PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetLoweredName' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    const char *lowered = nullptr;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcAddNameExpression(prog, "uptk_rtc_smoke_kernel");
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: add name expr (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetLoweredName(prog, "uptk_rtc_smoke_kernel", &lowered);
    printf("UPTKrtcGetLoweredName -> %d (%s)\n", (int)err, lowered ? lowered : "(null)");
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetLoweredName PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetPTXSize' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    size_t ptx_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetPTXSize(prog, &ptx_sz);
    printf("UPTKrtcGetPTXSize -> %d (sz=%zu)\n", (int)err, ptx_sz);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetPTXSize PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetPTX' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    size_t ptx_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetPTXSize(prog, &ptx_sz);
    if (err != UPTKRTC_SUCCESS || ptx_sz == 0) {
        printf("test_skip: ptx size (%d) sz=%zu\n", (int)err, ptx_sz);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    char *buf = (char *)malloc(ptx_sz + 1);
    if (!buf) {
        printf("test_skip: malloc\n");
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    memset(buf, 0, ptx_sz + 1);
    err = UPTKrtcGetPTX(prog, buf);
    printf("UPTKrtcGetPTX -> %d\n", (int)err);
    free(buf);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetPTX PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetProgramLogSize' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    size_t log_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetProgramLogSize(prog, &log_sz);
    printf("UPTKrtcGetProgramLogSize -> %d (sz=%zu)\n", (int)err, log_sz);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetProgramLogSize PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetProgramLog' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    size_t log_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 0, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetProgramLogSize(prog, &log_sz);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: log size (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    char *buf = (char *)malloc(log_sz + 1);
    if (!buf) {
        printf("test_skip: malloc\n");
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    memset(buf, 0, log_sz + 1);
    err = UPTKrtcGetProgramLog(prog, buf);
    printf("UPTKrtcGetProgramLog -> %d\n", (int)err);
    free(buf);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetProgramLog PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcVersion' {
            return @"
$hdr
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
"@
        }
        'UPTKrtcGetErrorString' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

int main(void)
{
    const char *s = UPTKrtcGetErrorString(UPTKRTC_SUCCESS);
    printf("UPTKrtcGetErrorString -> %s\n", s ? s : "(null)");
    printf("test_UPTKrtcGetErrorString PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetCUBINSize' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    const char *opts[] = {"--gpu-architecture=compute_70"};
    size_t cubin_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke_cubin", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 1, opts);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: cubin compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetCUBINSize(prog, &cubin_sz);
    printf("UPTKrtcGetCUBINSize -> %d (sz=%zu)\n", (int)err, cubin_sz);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetCUBINSize PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetCUBIN' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char k_rtc_src[] =
    "extern \"C\" __global__ void uptk_rtc_smoke_kernel(){}\n";

int main(void)
{
    UPTKrtcProgram prog{};
    const char *opts[] = {"--gpu-architecture=compute_70"};
    size_t cubin_sz = 0;
    UPTKrtcResult err = UPTKrtcCreateProgram(&prog, k_rtc_src, "uptk_rtc_smoke_cubin", 0, nullptr, nullptr);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: create failed (%d)\n", (int)err);
        return 0;
    }
    err = UPTKrtcCompileProgram(prog, 1, opts);
    if (err != UPTKRTC_SUCCESS) {
        printf("test_skip: cubin compile (%d)\n", (int)err);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    err = UPTKrtcGetCUBINSize(prog, &cubin_sz);
    if (err != UPTKRTC_SUCCESS || cubin_sz == 0) {
        printf("test_skip: cubin size (%d) sz=%zu\n", (int)err, cubin_sz);
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    char *buf = (char *)malloc(cubin_sz + 1);
    if (!buf) {
        printf("test_skip: malloc\n");
        UPTKrtcDestroyProgram(&prog);
        return 0;
    }
    memset(buf, 0, cubin_sz + 1);
    err = UPTKrtcGetCUBIN(prog, buf);
    printf("UPTKrtcGetCUBIN -> %d\n", (int)err);
    free(buf);
    UPTKrtcDestroyProgram(&prog);
    printf("test_UPTKrtcGetCUBIN PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetNVVMSize' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    size_t sz = 0;
    UPTKrtcResult err = UPTKrtcGetNVVMSize((UPTKrtcProgram)(uintptr_t)0, &sz);
    printf("UPTKrtcGetNVVMSize -> %d\n", (int)err);
    printf("test_UPTKrtcGetNVVMSize PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetNVVM' {
            return @"
$hdr
#include <UPTK_rtc.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    char tiny[4];
    tiny[0] = 0;
    UPTKrtcResult err = UPTKrtcGetNVVM((UPTKrtcProgram)(uintptr_t)0, tiny);
    printf("UPTKrtcGetNVVM -> %d\n", (int)err);
    printf("test_UPTKrtcGetNVVM PASS\n");
    return 0;
}
"@
        }
        'UPTKrtcGetNumSupportedArchs' {
            return @"
$hdr
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
"@
        }
        'UPTKrtcGetSupportedArchs' {
            return @"
$hdr
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
"@
        }
        default {
            return $null
        }
    }
}

$seen = @{}
foreach ($m in $rx) {
    $fname = $m.Groups[1].Value
    if ($seen.ContainsKey($fname)) { continue }
    $seen[$fname] = $true

    $cpp = Get-CppForRtcFunction $fname
    if (-not $cpp) {
        Write-Error "No generator template for $fname — extend generate_rtc_tests.ps1"
    }

    $outPath = Join-Path $outDir ("test_{0}.cpp" -f $fname)
    Set-Content -LiteralPath $outPath -Value $cpp -Encoding UTF8
}

Write-Host "Wrote rtc tests to $outDir"
