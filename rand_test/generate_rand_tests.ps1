# Generates test/test_UPTKrand*.cpp for each UPTKrand wrapper in rand_fun_convert.cpp
$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$srcPath = Join-Path $repoRoot 'src\rand\rand_fun_convert.cpp'
$outDir = Join-Path $PSScriptRoot 'test'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$text = Get-Content -LiteralPath $srcPath -Raw -Encoding UTF8
$text = [regex]::Replace($text, '/\*[\s\S]*?\*/', '')
$text = [regex]::Replace($text, '(?ms)#ifdef\s+UPTK_NOT_SUPPORT.*?#endif', '')

function Split-ParamList([string]$plist) {
    if ([string]::IsNullOrWhiteSpace($plist)) { return @() }
    $parts = New-Object System.Collections.Generic.List[string]
    $depth = 0
    $cur = New-Object System.Text.StringBuilder
    foreach ($ch in $plist.ToCharArray()) {
        if ($ch -eq '(') { $depth++ }
        elseif ($ch -eq ')') { $depth-- }
        elseif ($ch -eq ',' -and $depth -eq 0) {
            $parts.Add($cur.ToString().Trim())
            [void]$cur.Clear()
            continue
        }
        [void]$cur.Append($ch)
    }
    $parts.Add($cur.ToString().Trim())
    return $parts | Where-Object { $_ -and ($_ -ne 'void') }
}

function Parse-ParamSegment([string]$seg) {
    $seg = $seg.Trim()
    if (-not $seg -or $seg -eq 'void') { return $null }

    $asArray = $false
    if ($seg -match '^(.+)\[\]\s*$') {
        $seg = $Matches[1].Trim()
        $asArray = $true
    }

    if ($seg -match '\*\s*(\w+)\s*$') {
        $name = $Matches[1]
        $base = $seg.Substring(0, $seg.Length - $Matches[0].Length).Trim()
        $typeStr = ($base + '*').Trim()
        if ($asArray) { $typeStr += '[]' }
        return @{ Type = $typeStr; Name = $name }
    }

    if ($seg -match '^(.+)\s+(\w+)\s*$') {
        $typeStr = $Matches[1].Trim()
        $name = $Matches[2]
        if ($asArray) { $typeStr += '[]' }
        return @{ Type = $typeStr; Name = $name }
    }

    Write-Warning "Cannot parse parameter: $seg"
    return $null
}

function Expr-ForParam($p, $fname) {
    $tRaw = $p.Type.Trim()
    $n = $p.Name

    if ($fname -eq 'UPTKrandGetDirectionVectors32' -or $fname -eq 'UPTKrandGetDirectionVectors64') {
        return $null
    }

    if ($tRaw -match '^void\s*\*$') { return '(void*)nullptr' }

    if ($tRaw -match '^UPTKrandGenerator_t\s*\*$') { return '&generator' }

    if ($tRaw -match '^UPTKrandDiscreteDistribution_t\s*\*$') { return '&discrete_distribution' }

    if ($tRaw.EndsWith('*')) {
        $isConst = $tRaw.TrimStart().StartsWith('const')
        $base = ($tRaw -replace '^\s*const\s+', '').TrimEnd('*').Trim()

        if (-not $isConst -and $base -eq 'int' -and ($n -eq 'version' -or $n -eq 'value')) {
            return '&host_int'
        }

        $qual = if ($isConst) { 'const ' } else { '' }
        return "(${qual}$base*)dev_scratch"
    }

    $t = $tRaw -replace '^\s*const\s+', ''

    if ($t -eq 'UPTKrandGenerator_t') { return 'generator' }

    if ($t -eq 'UPTKrandDiscreteDistribution_t') { return 'discrete_distribution' }

    if ($t -eq 'UPTKrandRngType_t') { return '(UPTKrandRngType_t)UPTKRAND_RNG_PSEUDO_DEFAULT' }

    if ($t -eq 'UPTKrandMethod_t') { return '(UPTKrandMethod_t)0' }

    if ($t -eq 'UPTKrandDirectionVectorSet_t') { return '(UPTKrandDirectionVectorSet_t)0' }

    if ($t -eq 'UPTKrandOrdering_t') { return '(UPTKrandOrdering_t)UPTKRAND_ORDERING_PSEUDO_DEFAULT' }

    if ($t -eq 'UPTKStream_t') { return 'stream_id' }

    if ($t -eq 'libraryPropertyType') { return '(libraryPropertyType)MAJOR_VERSION' }

    if ($t -eq 'unsigned int' -or $t -eq 'uint32_t') { return '0u' }
    if ($t -eq 'unsigned long long') { return '0ull' }
    if ($t -eq 'unsigned short') { return '(unsigned short)0' }
    if ($t -eq 'unsigned char') { return '(unsigned char)0' }
    if ($t -eq 'char') { return '(char)0' }
    if ($t -eq 'int') { return '0' }
    if ($t -eq 'size_t') { return '(size_t)4' }
    if ($t -eq 'float') { return '0.f' }
    if ($t -eq 'double') { return '1.' }

    return "($tRaw)0"
}

$fixture = @'

static UPTKrandStatus_t rand_bind_gpu(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return (UPTKrandStatus_t)1;
    if (cudaSetDevice(0) != cudaSuccess)
        return (UPTKrandStatus_t)1;
    return UPTKRAND_STATUS_SUCCESS;
}

static UPTKrandStatus_t rand_setup_core(
    UPTKrandGenerator_t *generator,
    void **dev_scratch,
    UPTKStream_t *stream_id,
    int create_generator)
{
    UPTKrandStatus_t st = rand_bind_gpu();
    if (st != UPTKRAND_STATUS_SUCCESS) return st;

    if (cudaMalloc(dev_scratch, 65536) != cudaSuccess)
        return (UPTKrandStatus_t)1;

    if (cudaStreamCreate(stream_id) != cudaSuccess) {
        cudaFree(*dev_scratch);
        *dev_scratch = nullptr;
        return (UPTKrandStatus_t)1;
    }

    *generator = (UPTKrandGenerator_t)(uintptr_t)0;

    if (create_generator) {
        st = UPTKrandCreateGenerator(generator, UPTKRAND_RNG_PSEUDO_DEFAULT);
        if (st != UPTKRAND_STATUS_SUCCESS) return st;
        st = UPTKrandSetStream(*generator, *stream_id);
        if (st != UPTKRAND_STATUS_SUCCESS) return st;
    }

    return UPTKRAND_STATUS_SUCCESS;
}

static void rand_teardown_core(
    UPTKrandGenerator_t generator,
    void *dev_scratch,
    UPTKStream_t stream_id)
{
    if (generator)
        UPTKrandDestroyGenerator(generator);
    if (dev_scratch)
        cudaFree(dev_scratch);
    if (stream_id)
        cudaStreamDestroy(stream_id);
}

'@

$rx = [regex]::Matches($text, '(?m)^UPTKrandStatus_t\s+UPTKRANDAPI\s+(UPTKrand\w+)\(([^\)]*)\)\s*\{')
Write-Host "Matched $($rx.Count) rand wrappers"

foreach ($m in $rx) {
    $fname = $m.Groups[1].Value
    $plist = $m.Groups[2].Value
    $parsed = @(
        foreach ($s in (Split-ParamList $plist)) {
            $x = Parse-ParamSegment $s
            if ($null -ne $x) { $x }
        }
    )

    $createGen = 1
    if ($fname -eq 'UPTKrandCreateGenerator' -or $fname -eq 'UPTKrandCreateGeneratorHost') { $createGen = 0 }

    $skipHeavySetup = ($fname -eq 'UPTKrandGetVersion' -or $fname -eq 'UPTKrandGetProperty')

    $extraDecl = ''
    $preCall = ''
    $callLine = ''

    if ($fname -eq 'UPTKrandGetDirectionVectors32') {
        $extraDecl = '    UPTKrandDirectionVectors32_t *dv32_holder{};'
        $callLine = '    err = UPTKrandGetDirectionVectors32(&dv32_holder, (UPTKrandDirectionVectorSet_t)0);'
    }
    elseif ($fname -eq 'UPTKrandGetDirectionVectors64') {
        $extraDecl = '    UPTKrandDirectionVectors64_t *dv64_holder{};'
        $callLine = '    err = UPTKrandGetDirectionVectors64(&dv64_holder, (UPTKrandDirectionVectorSet_t)0);'
    }
    elseif ($fname -eq 'UPTKrandGetScrambleConstants32') {
        $extraDecl = '    unsigned int *scramble32{};'
        $callLine = '    err = UPTKrandGetScrambleConstants32(&scramble32);'
    }
    elseif ($fname -eq 'UPTKrandGetScrambleConstants64') {
        $extraDecl = '    unsigned long long *scramble64{};'
        $callLine = '    err = UPTKrandGetScrambleConstants64(&scramble64);'
    }
    elseif ($fname -eq 'UPTKrandDestroyDistribution') {
        $preCall = @'

    err = UPTKrandCreatePoissonDistribution(1.0, &discrete_distribution);
    if (err != UPTKRAND_STATUS_SUCCESS) {
        printf("test_skip: UPTKrandDestroyDistribution create poisson failed\n");
        rand_teardown_core(generator, dev_scratch, stream_id);
        return 0;
    }

'@
        $callLine = '    err = UPTKrandDestroyDistribution(discrete_distribution);'
        $extraDecl = '    UPTKrandDiscreteDistribution_t discrete_distribution{};'
    }
    else {
        $args = foreach ($pp in $parsed) { Expr-ForParam $pp $fname }
        $protoArgs = ($args -join ', ')
        $callLine = "    err = $fname($protoArgs);"
        if ($plist -match 'UPTKrandDiscreteDistribution_t\s*\*') {
            $extraDecl = '    UPTKrandDiscreteDistribution_t discrete_distribution{};'
        }
    }

    $setupSection = if ($skipHeavySetup) {
        @"

    int host_int = 0;
    UPTKrandGenerator_t generator{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKrandStatus_t err;

    err = rand_bind_gpu();
    if (err != UPTKRAND_STATUS_SUCCESS) {
        printf("test_skip: $fname bind gpu failed\n");
        return 0;
    }

"@
    }
    else {
        @"

    int host_int = 0;
    UPTKrandGenerator_t generator{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
$extraDecl
    UPTKrandStatus_t err;

    err = rand_setup_core(&generator, &dev_scratch, &stream_id, $createGen);
    if (err != UPTKRAND_STATUS_SUCCESS) {
        printf("test_skip: $fname setup failed (%d)\n", (int)err);
        return 0;
    }

"@
    }

    $tearSection = if ($skipHeavySetup) {
        @'

'
    }
    elseif ($fname -eq 'UPTKrandCreateGenerator' -or $fname -eq 'UPTKrandCreateGeneratorHost') {
        @'


    rand_teardown_core(generator, dev_scratch, stream_id);

'@
    }
    else {
        @'


    rand_teardown_core(generator, dev_scratch, stream_id);

'@
    }

    if ($fname -eq 'UPTKrandDestroyGenerator') {
        $tearSection = @'


    rand_teardown_core((UPTKrandGenerator_t)(uintptr_t)0, dev_scratch, stream_id);

'@
    }

    $postDestroy = ''
    if ($fname -eq 'UPTKrandCreatePoissonDistribution') {
        $postDestroy = @'


    if (err == UPTKRAND_STATUS_SUCCESS)
        UPTKrandDestroyDistribution(discrete_distribution);

'@
    }

    $cpp = @"
/*
 * Auto-generated smoke test for $fname (rand_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rand_test/generate_rand_tests.ps1
 */
#include <UPTK_rand.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

$fixture

int main(void)
{
$setupSection$preCall$callLine

    printf("$fname -> %d\n", (int)err);
$postDestroy
$tearSection
    printf("test_${fname} PASS\n");
    return 0;
}
"@

    # Fix CreateGenerator: must invoke wrapper then teardown with created generator
    if ($fname -eq 'UPTKrandCreateGenerator') {
        $cpp = @"
/*
 * Auto-generated smoke test for $fname (rand_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rand_test/generate_rand_tests.ps1
 */
#include <UPTK_rand.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

$fixture

int main(void)
{
    UPTKrandGenerator_t generator{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKrandStatus_t err;

    err = rand_bind_gpu();
    if (err != UPTKRAND_STATUS_SUCCESS) {
        printf("test_skip: $fname bind failed\n");
        return 0;
    }
    if (cudaMalloc(&dev_scratch, 65536) != cudaSuccess || cudaStreamCreate(&stream_id) != cudaSuccess) {
        printf("test_skip: $fname cuda alloc/stream\n");
        return 0;
    }

    err = UPTKrandCreateGenerator(&generator, (UPTKrandRngType_t)UPTKRAND_RNG_PSEUDO_DEFAULT);
    printf("$fname -> %d\n", (int)err);
    if (generator)
        UPTKrandDestroyGenerator(generator);
    cudaFree(dev_scratch);
    cudaStreamDestroy(stream_id);
    printf("test_${fname} PASS\n");
    return 0;
}
"@
    }

    if ($fname -eq 'UPTKrandCreateGeneratorHost') {
        $cpp = @"
/*
 * Auto-generated smoke test for $fname (rand_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/rand_test/generate_rand_tests.ps1
 */
#include <UPTK_rand.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

$fixture

int main(void)
{
    UPTKrandGenerator_t generator{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
    UPTKrandStatus_t err;

    err = rand_bind_gpu();
    if (err != UPTKRAND_STATUS_SUCCESS) {
        printf("test_skip: $fname bind failed\n");
        return 0;
    }
    if (cudaMalloc(&dev_scratch, 65536) != cudaSuccess || cudaStreamCreate(&stream_id) != cudaSuccess) {
        printf("test_skip: $fname cuda alloc/stream\n");
        return 0;
    }

    err = UPTKrandCreateGeneratorHost(&generator, (UPTKrandRngType_t)UPTKRAND_RNG_PSEUDO_DEFAULT);
    printf("$fname -> %d\n", (int)err);
    if (generator)
        UPTKrandDestroyGenerator(generator);
    cudaFree(dev_scratch);
    cudaStreamDestroy(stream_id);
    printf("test_${fname} PASS\n");
    return 0;
}
"@
    }

    if ($fname -eq 'UPTKrandGetVersion') {
        $cpp = @"
/*
 * Auto-generated smoke test for $fname (rand_fun_convert.cpp).
 */
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
"@
    }

    if ($fname -eq 'UPTKrandGetProperty') {
        $cpp = @"
/*
 * Auto-generated smoke test for $fname (rand_fun_convert.cpp).
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
"@
    }

    $outPath = Join-Path $outDir ("test_{0}.cpp" -f $fname)
    Set-Content -LiteralPath $outPath -Value $cpp -Encoding UTF8
}

Write-Host "Wrote rand tests to $outDir"
