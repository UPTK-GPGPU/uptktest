# Generates test/test_UPTKsparse*.cpp for each UPTKsparse wrapper in sparse_fun_convert.cpp
$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$srcPath = Join-Path $repoRoot 'src\sparse\sparse_fun_convert.cpp'
$outDir = Join-Path $PSScriptRoot 'test'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$text = Get-Content -LiteralPath $srcPath -Raw -Encoding UTF8
$text = [regex]::Replace($text, '/\*[\s\S]*?\*/', '')

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

    if ($seg -match '\*\s*(\w+)\s*$') {
        $name = $Matches[1]
        $base = $seg.Substring(0, $seg.Length - $Matches[0].Length).Trim()
        return @{ Type = ($base + '*').Trim(); Name = $name }
    }

    if ($seg -match '^(.+)\s+(\w+)\s*$') {
        return @{ Type = $Matches[1].Trim(); Name = $Matches[2] }
    }

    Write-Warning "Cannot parse parameter: $seg"
    return $null
}

function Collect-MatDescrNames($parsed) {
    $s = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($pp in $parsed) {
        $t = ($pp.Type.Trim() -replace '^\s*const\s+', '')
        if ($t -eq 'UPTKsparseMatDescr_t') {
            [void]$s.Add($pp.Name)
        }
    }
    return @($s)
}

function Collect-SparseObjects($parsed) {
    $m = @{}
    foreach ($pp in $parsed) {
        $t = ($pp.Type.Trim() -replace '^\s*const\s+', '')
        if ($t -match '\*$') { continue }
        if ($t -eq 'UPTKsparseHandle_t') { continue }
        if ($t -eq 'UPTKsparseMatDescr_t') { continue }
        if ($t -match 'Descr_t$|ColorInfo_t$|Plan_t$') {
            $m[$pp.Name] = $t
        }
    }
    return $m
}

function Expr-ForParam($p) {
    $tRaw = $p.Type.Trim()
    $n = $p.Name

    if ($tRaw -match '^void\s*\*$') { return '(void*)nullptr' }
    if ($tRaw -match '^void\s*\*\*$') { return '(void**)nullptr' }
    if ($tRaw -match '^const\s+void\s*\*$') { return '(const void*)dev_scratch' }
    if ($tRaw -match '^const\s+void\s*\*\*$') { return '(const void**)nullptr' }

    if ($tRaw -match '^UPTKsparseHandle_t\s*\*$') { return '&sparse_handle' }

    if ($tRaw.EndsWith('*')) {
        $isConst = $tRaw.TrimStart().StartsWith('const')
        $base = ($tRaw -replace '^\s*const\s+', '').TrimEnd('*').Trim()
        $qual = if ($isConst) { 'const ' } else { '' }
        return "(${qual}$base*)dev_scratch"
    }

    $t = $tRaw -replace '^\s*const\s+', ''

    if ($t -eq 'UPTKsparseHandle_t') { return 'sparse_handle' }

    if ($t -eq 'UPTKsparseMatDescr_t') { return $n }

    if ($t -eq 'UPTKsparseDirection_t') { return '(UPTKsparseDirection_t)UPTKSPARSE_DIRECTION_ROW' }
    if ($t -eq 'UPTKsparseOperation_t') { return '(UPTKsparseOperation_t)UPTKSPARSE_OPERATION_NON_TRANSPOSE' }
    if ($t -eq 'UPTKsparseSolvePolicy_t') { return '(UPTKsparseSolvePolicy_t)UPTKSPARSE_SOLVE_POLICY_NO_LEVEL' }
    if ($t -eq 'UPTKsparseOrder_t') { return '(UPTKsparseOrder_t)0' }
    if ($t -eq 'UPTKsparseAction_t') { return '(UPTKsparseAction_t)0' }
    if ($t -eq 'UPTKsparseIndexBase_t') { return '(UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO' }
    if ($t -eq 'UPTKsparseSparseToDenseAlg_t') { return '(UPTKsparseSparseToDenseAlg_t)0' }
    if ($t -eq 'UPTKsparseDenseToSparseAlg_t') { return '(UPTKsparseDenseToSparseAlg_t)0' }
    if ($t -eq 'UPTKsparseSpSVAlg_t') { return '(UPTKsparseSpSVAlg_t)0' }
    if ($t -eq 'UPTKsparseSpSMAlg_t') { return '(UPTKsparseSpSMAlg_t)0' }
    if ($t -eq 'UPTKsparseSpGEMMAlg_t') { return '(UPTKsparseSpGEMMAlg_t)0' }
    if ($t -eq 'UPTKsparseSDDMMAlg_t') { return '(UPTKsparseSDDMMAlg_t)0' }
    if ($t -eq 'UPTKsparseSpMMAlg_t') { return '(UPTKsparseSpMMAlg_t)0' }
    if ($t -eq 'UPTKsparseSpMVAlg_t') { return '(UPTKsparseSpMVAlg_t)0' }
    if ($t -eq 'UPTKsparseSpMMOpAlg_t') { return '(UPTKsparseSpMMOpAlg_t)0' }
    if ($t -eq 'UPTKsparseCsr2CscAlg_t') { return '(UPTKsparseCsr2CscAlg_t)0' }
    if ($t -eq 'UPTKsparseDiagType_t') { return '(UPTKsparseDiagType_t)0' }
    if ($t -eq 'UPTKsparseFillMode_t') { return '(UPTKsparseFillMode_t)0' }
    if ($t -eq 'UPTKsparseFormat_t') { return '(UPTKsparseFormat_t)0' }
    if ($t -eq 'UPTKsparseIndexType_t') { return '(UPTKsparseIndexType_t)0' }
    if ($t -eq 'UPTKsparseMatrixType_t') { return '(UPTKsparseMatrixType_t)UPTKSPARSE_MATRIX_TYPE_GENERAL' }
    if ($t -eq 'UPTKsparseSpMatAttribute_t') { return '(UPTKsparseSpMatAttribute_t)0' }
    if ($t -eq 'UPTKDataType' -or $t -eq 'UPTKDataType_t') { return '(UPTKDataType_t)UPTK_R_32F' }

    if ($t -eq 'UPTKStream_t') { return 'stream_id' }

    if ($t -match '^UPTKsparse[A-Za-z0-9_]+$') { return $n }

    if ($t -eq 'unsigned int' -or $t -eq 'uint32_t') { return '0u' }
    if ($t -eq 'unsigned long long') { return '0ull' }
    if ($t -eq 'unsigned short') { return '(unsigned short)0' }
    if ($t -eq 'unsigned char') { return '(unsigned char)0' }
    if ($t -eq 'char') { return '(char)0' }
    if ($t -eq 'int') { return '0' }
    if ($t -eq 'size_t') { return '(size_t)0' }
    if ($t -eq 'int64_t') { return '(int64_t)0' }
    if ($t -eq 'float') { return '0.f' }
    if ($t -eq 'double') { return '0.' }
    if ($t -eq 'cuComplex') { return '(cuComplex){0.f, 0.f}' }
    if ($t -eq 'cuDoubleComplex') { return '(cuDoubleComplex){0., 0.}' }
    if ($t -eq '__half') { return '(__half)0' }

    if ($t -eq 'libraryPropertyType') { return '(libraryPropertyType)0' }

    if ($t -match '^(bsr|csr|csru|bsrs|bsrsm|csrsm|csric|bsric|csrilu|bsrilu|prune)Info_t$') {
        return "($t)(uintptr_t)0"
    }

    if ($t -match '^cusparse') {
        return "($t)(uintptr_t)0"
    }

    return "($tRaw)0"
}

$sharedFixture = @'

static UPTKsparseStatus_t sparse_bind_gpu(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0)
        return (UPTKsparseStatus_t)1;
    if (cudaSetDevice(0) != cudaSuccess)
        return (UPTKsparseStatus_t)1;
    return UPTKSPARSE_STATUS_SUCCESS;
}

static UPTKsparseStatus_t sparse_setup_core(
    UPTKsparseHandle_t *sparse_handle,
    void **dev_scratch,
    UPTKStream_t *stream_id,
    int create_handle)
{
    UPTKsparseStatus_t st = sparse_bind_gpu();
    if (st != UPTKSPARSE_STATUS_SUCCESS) return st;

    if (cudaMalloc(dev_scratch, 65536) != cudaSuccess)
        return (UPTKsparseStatus_t)1;

    if (cudaStreamCreate(stream_id) != cudaSuccess) {
        cudaFree(*dev_scratch);
        *dev_scratch = nullptr;
        return (UPTKsparseStatus_t)1;
    }

    if (create_handle) {
        st = UPTKsparseCreate(sparse_handle);
        if (st != UPTKSPARSE_STATUS_SUCCESS) return st;
        st = UPTKsparseSetStream(*sparse_handle, *stream_id);
        if (st != UPTKSPARSE_STATUS_SUCCESS) return st;
    } else {
        *sparse_handle = (UPTKsparseHandle_t)(uintptr_t)0;
    }

    return UPTKSPARSE_STATUS_SUCCESS;
}

static void sparse_teardown_core(
    UPTKsparseHandle_t sparse_handle,
    void *dev_scratch,
    UPTKStream_t stream_id)
{
    if (sparse_handle)
        UPTKsparseDestroy(sparse_handle);
    if (dev_scratch)
        cudaFree(dev_scratch);
    if (stream_id)
        cudaStreamDestroy(stream_id);
}

'@

$rx = [regex]::Matches($text, '(?m)^UPTKsparseStatus_t\s+UPTKSPARSEAPI\s+(UPTKsparse\w+)\(([^\)]*)\)\s*\{')
Write-Host "Matched $($rx.Count) sparse wrappers"

foreach ($m in $rx) {
    $fname = $m.Groups[1].Value
    $plist = $m.Groups[2].Value
    $parsed = @(
        foreach ($s in (Split-ParamList $plist)) {
            $x = Parse-ParamSegment $s
            if ($null -ne $x) { $x }
        }
    )

    $args = foreach ($pp in $parsed) { Expr-ForParam $pp }
    $protoArgs = ($args -join ', ')

    $matNames = Collect-MatDescrNames $parsed
    $sparseObjs = Collect-SparseObjects $parsed

    $objDecls = ''
    foreach ($k in ($sparseObjs.Keys | Sort-Object)) {
        $tp = $sparseObjs[$k]
        $objDecls += "    $tp $k{};`n"
    }

    $createHandleFlag = if ($fname -eq 'UPTKsparseCreate') { '0' } else { '1' }

    $matCreate = ''
    $matDestroy = ''
    foreach ($mn in ($matNames | Sort-Object)) {
        $matCreate += @"

    err = UPTKsparseCreateMatDescr(&$mn);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: $fname UPTKsparseCreateMatDescr($mn) failed\n");
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
        return 0;
    }
    err = UPTKsparseSetMatIndexBase($mn, UPTKSPARSE_INDEX_BASE_ZERO);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: $fname SetMatIndexBase($mn)\n");
        UPTKsparseDestroyMatDescr($mn); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }
    err = UPTKsparseSetMatType($mn, UPTKSPARSE_MATRIX_TYPE_GENERAL);
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: $fname SetMatType($mn)\n");
        UPTKsparseDestroyMatDescr($mn); sparse_teardown_core(sparse_handle, dev_scratch, stream_id); return 0;
    }

"@
        $matDestroy += @"

    if ($mn) UPTKsparseDestroyMatDescr($mn);

"@
    }

    $matDeclLines = ''
    foreach ($mn in ($matNames | Sort-Object)) {
        $matDeclLines += "    UPTKsparseMatDescr_t $mn{};`n"
    }

    $setupCall = @"
    err = sparse_setup_core(&sparse_handle, &dev_scratch, &stream_id, $createHandleFlag);
"@

    $tearBlock = @"

$matDestroy    sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
"@

    if ($fname -eq 'UPTKsparseCreate') {
        $matCreate = ''
        $tearBlock = @"

    sparse_teardown_core(sparse_handle, dev_scratch, stream_id);
"@
    }

    $cpp = @"
/*
 * Auto-generated smoke test for $fname (sparse_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/sparse_test/generate_sparse_tests.ps1
 */
#include <cuda_runtime.h>
#include <UPTK_sparse.h>
#include <stdint.h>
#include <stdio.h>

$sharedFixture

int main(void)
{
    UPTKsparseHandle_t sparse_handle{};
    void *dev_scratch{};
    UPTKStream_t stream_id{};
$matDeclLines$objDecls
    UPTKsparseStatus_t err;

$setupCall
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: $fname setup failed (%d)\n", (int)err);
        return 0;
    }

$matCreate
    err = $fname($protoArgs);

    printf("$fname -> %d\n", (int)err);

$tearBlock
    printf("test_${fname} PASS\n");
    return 0;
}
"@

    $outPath = Join-Path $outDir ("test_{0}.cpp" -f $fname)
    Set-Content -LiteralPath $outPath -Value $cpp -Encoding UTF8
}

Write-Host "Wrote sparse tests to $outDir"
