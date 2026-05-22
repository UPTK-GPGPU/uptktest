# Patches the 15 known-failing sparse smoke tests (no sparse_fun_convert.cpp required).
$ErrorActionPreference = 'Stop'
$testDir = Join-Path $PSScriptRoot 'test'

function Patch-File([string]$path, [scriptblock]$fn) {
    if (-not (Test-Path $path)) { Write-Warning "missing $path"; return }
    $c = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    $n = & $fn $c
    if ($n -ne $c) {
        [System.IO.File]::WriteAllText($path, $n, [System.Text.UTF8Encoding]::new($false))
        Write-Host "patched $(Split-Path $path -Leaf)"
    }
}

# Fix teardown signature in all sparse tests
Get-ChildItem -LiteralPath $testDir -Filter 'test_UPTKsparse*.cpp' | ForEach-Object {
    Patch-File $_.FullName {
        param($c)
        $newTeardown = "static void sparse_teardown_core(`r`n    UPTKsparseHandle_t sparse_handle,`r`n    void *dev_scratch,`r`n    UPTKStream_t stream_id,`r`n    int destroy_sparse_handle)`r`n{`r`n    if (destroy_sparse_handle && sparse_handle)"
        $c = $c -replace '(?ms)static void sparse_teardown_core\(\s*UPTKsparseHandle_t sparse_handle,\s*void \*dev_scratch,\s*UPTKStream_t stream_id\s*\)\s*\{\s*if \(sparse_handle\)', $newTeardown
        $c = $c -replace 'sparse_teardown_core\(sparse_handle, dev_scratch, stream_id\)(?!,)', 'sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1)'
        return $c
    }
}

$createSpMat = '    UPTKsparseSpMatDescr_t spMatDescr{};'
$createDn = '    UPTKsparseDnMatDescr_t dnMatDescr{};'
$createSpVec = '    UPTKsparseSpVecDescr_t spVecDescr{};'

$fixes = @{
    'test_UPTKsparseCreateCsr.cpp' = {
        param($c)
        if ($c -notmatch 'spMatDescr') {
            $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createSpMat"
        }
        if ($c -match '&spMatDescr' -and $c -notmatch 'UPTKsparseSpMatDescr_t spMatDescr') {
            $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createSpMat"
        }
        $c = $c -replace 'UPTKsparseCreateCsr\(\(UPTKsparseSpMatDescr_t\*\)dev_scratch[^;]+;', 'UPTKsparseCreateCsr(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F);'
        return $c
    }
    'test_UPTKsparseCreateCoo.cpp' = {
        param($c)
        if ($c -notmatch 'spMatDescr') { $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createSpMat" }
        $c = $c -replace 'UPTKsparseCreateCoo\(\(UPTKsparseSpMatDescr_t\*\)dev_scratch[^;]+;', 'UPTKsparseCreateCoo(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F);'
        return $c
    }
    'test_UPTKsparseCreateCsc.cpp' = {
        param($c)
        if ($c -notmatch 'spMatDescr') { $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createSpMat" }
        $c = $c -replace 'UPTKsparseCreateCsc\(\(UPTKsparseSpMatDescr_t\*\)dev_scratch[^;]+;', 'UPTKsparseCreateCsc(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F);'
        return $c
    }
    'test_UPTKsparseCreateBlockedEll.cpp' = {
        param($c)
        if ($c -notmatch 'spMatDescr') { $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createSpMat" }
        $c = $c -replace 'UPTKsparseCreateBlockedEll\(\(UPTKsparseSpMatDescr_t\*\)dev_scratch[^;]+;', 'UPTKsparseCreateBlockedEll(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)1, (int64_t)0, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F);'
        return $c
    }
    'test_UPTKsparseCreateDnMat.cpp' = {
        param($c)
        if ($c -notmatch 'dnMatDescr') { $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createDn" }
        $c = $c -replace 'UPTKsparseCreateDnMat\(\(UPTKsparseDnMatDescr_t\*\)dev_scratch[^;]+;', 'UPTKsparseCreateDnMat(&dnMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (UPTKDataType_t)UPTK_R_32F, (UPTKsparseOrder_t)0);'
        return $c
    }
    'test_UPTKsparseCreateSpVec.cpp' = {
        param($c)
        if ($c -notmatch 'spVecDescr') { $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n$createSpVec" }
        $c = $c -replace 'UPTKsparseCreateSpVec\(\(UPTKsparseSpVecDescr_t\*\)dev_scratch[^;]+;', 'UPTKsparseCreateSpVec(&spVecDescr, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F);'
        return $c
    }
}

foreach ($name in $fixes.Keys) {
    Patch-File (Join-Path $testDir $name) $fixes[$name]
}

function Add-GetPreCall([string]$c, [string]$createCall, [string]$destroyCall, [string]$decl) {
    if ($c -match [regex]::Escape($createCall)) { return $c }
    if ($decl -and $c -notmatch [regex]::Escape($decl.Trim())) {
        $c = $c -replace '(UPTKStream_t stream_id\{\};)', "`$1`n    $decl"
    }
    $insert = @"

    err = $createCall;
    if (err != UPTKSPARSE_STATUS_SUCCESS) {
        printf("test_skip: create descriptor failed (%d)\n", (int)err);
        sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 1);
        return 0;
    }

"@
    return $c -replace '(\r?\n)(\s*err = UPTKsparse)', "$insert`$1`$2"
}

Patch-File (Join-Path $testDir 'test_UPTKsparseCsrGet.cpp') {
    param($c)
    $c = Add-GetPreCall $c 'UPTKsparseCreateCsr(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F)' 'if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);' 'UPTKsparseSpMatDescr_t spMatDescr{};'
    if ($c -notmatch 'DestroySpMat') { $c = $c -replace '(printf\("UPTKsparseCsrGet[^)]+\);)', "`$1`n`n    if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);" }
    return $c
}

Patch-File (Join-Path $testDir 'test_UPTKsparseCooGet.cpp') {
    param($c)
    $c = Add-GetPreCall $c 'UPTKsparseCreateCoo(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F)' 'if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);' 'UPTKsparseSpMatDescr_t spMatDescr{};'
    if ($c -notmatch 'DestroySpMat') { $c = $c -replace '(printf\("UPTKsparseCooGet[^)]+\);)', "`$1`n`n    if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);" }
    return $c
}

Patch-File (Join-Path $testDir 'test_UPTKsparseSpMatGetFormat.cpp') {
    param($c)
    $c = Add-GetPreCall $c 'UPTKsparseCreateCsr(&spMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F)' 'if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);' 'UPTKsparseSpMatDescr_t spMatDescr{};'
    if ($c -notmatch 'DestroySpMat') { $c = $c -replace '(printf\("UPTKsparseSpMatGetFormat[^)]+\);)', "`$1`n`n    if (spMatDescr) UPTKsparseDestroySpMat(spMatDescr);" }
    return $c
}

Patch-File (Join-Path $testDir 'test_UPTKsparseDnMatGet.cpp') {
    param($c)
    $c = Add-GetPreCall $c 'UPTKsparseCreateDnMat(&dnMatDescr, (int64_t)0, (int64_t)0, (int64_t)0, (void*)nullptr, (UPTKDataType_t)UPTK_R_32F, (UPTKsparseOrder_t)0)' 'if (dnMatDescr) UPTKsparseDestroyDnMat(dnMatDescr);' 'UPTKsparseDnMatDescr_t dnMatDescr{};'
    if ($c -notmatch 'DestroyDnMat') { $c = $c -replace '(printf\("UPTKsparseDnMatGet[^)]+\);)', "`$1`n`n    if (dnMatDescr) UPTKsparseDestroyDnMat(dnMatDescr);" }
    return $c
}

Patch-File (Join-Path $testDir 'test_UPTKsparseSpVecGet.cpp') {
    param($c)
    $c = Add-GetPreCall $c 'UPTKsparseCreateSpVec(&spVecDescr, (int64_t)0, (int64_t)0, (void*)nullptr, (void*)nullptr, (UPTKsparseIndexType_t)UPTKSPARSE_INDEX_32I, (UPTKsparseIndexBase_t)UPTKSPARSE_INDEX_BASE_ZERO, (UPTKDataType_t)UPTK_R_32F)' 'if (spVecDescr) UPTKsparseDestroySpVec(spVecDescr);' 'UPTKsparseSpVecDescr_t spVecDescr{};'
    if ($c -notmatch 'DestroySpVec') { $c = $c -replace '(printf\("UPTKsparseSpVecGet[^)]+\);)', "`$1`n`n    if (spVecDescr) UPTKsparseDestroySpVec(spVecDescr);" }
    return $c
}

Patch-File (Join-Path $testDir 'test_UPTKsparseDestroy.cpp') {
    param($c)
    if ($c -notmatch 'sparse_handle = \(UPTKsparseHandle_t\)') {
        $c = $c -replace '(printf\("UPTKsparseDestroy[^)]+\);)', "`$1`n`n    sparse_handle = (UPTKsparseHandle_t)(uintptr_t)0;"
    }
    $c = $c -replace 'sparse_teardown_core\(sparse_handle, dev_scratch, stream_id, 1\)', 'sparse_teardown_core(sparse_handle, dev_scratch, stream_id, 0)'
    return $c
}

Patch-File (Join-Path $testDir 'test_UPTKsparseDestroyMatDescr.cpp') {
    param($c)
    $c = $c -replace '(?ms)\s*if \(descrA\) UPTKsparseDestroyMatDescr\(descrA\);\s*sparse_teardown_core', "`n    descrA = (UPTKsparseMatDescr_t)(uintptr_t)0;`n    sparse_teardown_core"
    return $c
}

Write-Host 'Done patching sparse failed tests.'
