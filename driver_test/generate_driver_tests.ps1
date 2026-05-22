# Generates test/test_UP*.cpp for each UPTKError UP*(...) in driver_fun_convert.cpp
$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$srcPath = Join-Path $repoRoot 'src\driver\driver_fun_convert.cpp'
$outDir = Join-Path $PSScriptRoot 'test'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$text = Get-Content -LiteralPath $srcPath -Raw -Encoding UTF8
$text = [regex]::Replace($text, '/\*[\s\S]*?\*/', '')
# Strip CUDA 12+ only wrappers -- they won't link on CUDA 11
$text = [regex]::Replace($text, '(?ms)#if\s+defined\(CUDA_VERSION\)\s*&&\s*CUDA_VERSION\s*>=\s*12000.*?#endif\s*/\*\s*CUDA_VERSION\s*>=\s*12000\s*\*/', '')

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
    if ($seg -match '^(.+)\s+(\w+)\s*$') {
        return @{ Type = $Matches[1].Trim(); Name = $Matches[2] }
    }
    Write-Warning "Cannot parse parameter: $seg"
    return $null
}

function Needs-GlTypes([string]$plist) {
    return ($plist -match '\bGLuint\b') -or ($plist -match '\bGLenum\b')
}

function Get-GraphLevel([string]$fname, $parsed) {
    if ($fname -eq 'UPGraphCreate') { return -1 }
    $hasGraphType = $false
    foreach ($p in $parsed) {
        $b = ($p.Type.TrimEnd('*')).Trim()
        if ($b -like 'UPTKGraph*') { $hasGraphType = $true }
    }
    if (-not $hasGraphType) { return -1 }

    $lvl = -1
    foreach ($p in $parsed) {
        $ts = $p.Type.Trim()
        if ($ts.EndsWith('*')) { continue }
        $t = $ts
        if ($t -eq 'UPTKGraphExec_t') { return 2 }
        if ($t -eq 'UPTKGraphNode_t') { $lvl = [Math]::Max($lvl, 1) }
        if ($t -eq 'UPTKGraph_t') { $lvl = [Math]::Max($lvl, 0) }
    }
    return $lvl
}

function Needs-Event($parsed) {
    foreach ($p in $parsed) {
        if ($p.Type.Trim() -eq 'UPTKEvent_t') { return $true }
    }
    return $false
}

function Needs-Array($parsed) {
    foreach ($p in $parsed) {
        if ($p.Type.Trim() -eq 'UPTKarray') { return $true }
    }
    return $false
}

function Needs-Mipmap($parsed) {
    foreach ($p in $parsed) {
        if ($p.Type.Trim() -eq 'UPTKMipmappedArray_t') { return $true }
    }
    return $false
}

function Needs-TexRef($parsed) {
    foreach ($p in $parsed) {
        if ($p.Type.Trim() -eq 'UPTKtexref') { return $true }
    }
    return $false
}

function Expr-ForParam($p) {
    $tRaw = $p.Type.Trim()
    $n = $p.Name

    if ($tRaw -match '^void\s*\*$') { return '(void*)nullptr' }
    if ($tRaw -match '^const\s+void\s*\*$') { return '(const void*)nullptr' }
    if ($tRaw -match '^void\s*\*\*$') { return '(void**)nullptr' }
    if ($tRaw -match '^const\s+void\s*\*\*$') { return '(const void**)nullptr' }

    if ($tRaw.EndsWith('*')) {
        return "&local_$n"
    }

    $t = $tRaw -replace '^\s*const\s+', ''

    if ($t -eq 'UPTKdevice') { return 'dev' }
    if ($t -eq 'UPTKcontext') { return 'ctx' }
    if ($t -eq 'UPTKStream_t') { return 'stream' }
    if ($t -eq 'UPTKEvent_t') { return 'evt' }
    if ($t -eq 'UPTKGraph_t') { return 'graph' }
    if ($t -eq 'UPTKGraphExec_t') { return 'graphExec' }
    if ($t -eq 'UPTKGraphNode_t') { return 'graphNode' }
    if ($t -eq 'UPTKarray') { return 'arr' }
    if ($t -eq 'UPTKMipmappedArray_t') { return 'mipmap' }
    if ($t -eq 'UPTKdeviceptr') { return 'dptr' }
    if ($t -eq 'UPTKmodule') { return 'mod' }
    if ($t -eq 'UPTKfunction') { return 'kern' }
    if ($t -eq 'UPTKtexref') { return 'texRef' }
    if ($t -eq 'UPTKsurfref') { return 'surfRef' }
    if ($t -eq 'UPTKExternalMemory_t') { return 'extMem' }
    if ($t -eq 'UPTKExternalSemaphore_t') { return 'extSem' }
    if ($t -eq 'UPTKGraphicsResource_t') { return 'gfxRes' }
    if ($t -eq 'UPTKSurfaceObject_t') { return 'surfObj' }
    if ($t -eq 'UPTKTextureObject_t') { return 'texObj' }
    if ($t -eq 'UPTKMemPool_t') { return 'memPool' }
    if ($t -eq 'UPTKUserObject_t') { return 'userObj' }
    if ($t -eq 'UPTKlinkState') { return 'linkState' }

    if ($t -eq 'UPTKhostFn') { return '(UPTKhostFn)nullptr' }
    if ($t -eq 'UPTKoccupancyB2DSize') { return '(UPTKoccupancyB2DSize)nullptr' }
    if ($t -eq 'UPTKstreamCallback') { return '(UPTKstreamCallback)nullptr' }

    if ($t -eq 'unsigned int' -or $t -eq 'uint32_t') { return '0u' }
    if ($t -eq 'unsigned long long') { return '0ull' }
    if ($t -eq 'unsigned short') { return '(unsigned short)0' }
    if ($t -eq 'unsigned char') { return '(unsigned char)0' }
    if ($t -eq 'int') { return '0' }
    if ($t -eq 'size_t') { return '(size_t)0' }
    if ($t -eq 'float') { return '0.f' }
    if ($t -eq 'double') { return '0.' }
    if ($t -eq 'bool') { return 'false' }
    if ($t -eq 'cuuint32_t') { return '(cuuint32_t)0' }
    if ($t -eq 'cuuint64_t') { return '(cuuint64_t)0' }

    if ($t -eq 'GLuint' -or $t -eq 'GLenum') { return '(GLuint)0' }

    if ($t -match '^char\s*\*$') { return '(char*)dummy_pci_bus_id' }

    if ($t -eq 'void *') { return '(void*)nullptr' }
    if ($t -eq 'void **') { return '(void**)nullptr' }
    if ($t -eq 'const void *') { return '(const void*)nullptr' }

    if ($t -eq 'UPTKlimit') { return '(UPTKlimit)UPTK_LIMIT_STACK_SIZE' }
    if ($t -eq 'UPTKfunc_cache') { return '(UPTKfunc_cache)0' }
    if ($t -eq 'UPTKsharedconfig') { return '(UPTKsharedconfig)0' }
    if ($t -eq 'UPTKStreamCaptureMode') { return '(UPTKStreamCaptureMode)0' }
    if ($t -eq 'UPTKStreamCaptureStatus') { return 'captureStatus' }

    if ($t -match '^UPTK[A-Za-z0-9_]+$') { return "($t){}" }

    return "($tRaw){}"
}

function LocalDecl-ForParam($p) {
    $tRaw = $p.Type.Trim()
    if (-not $tRaw.EndsWith('*')) { return $null }
    if ($tRaw -match '^void\s*\*$') { return $null }
    if ($tRaw -match '^const\s+void\s*\*$') { return $null }
    if ($tRaw -match '^void\s*\*\*$') { return $null }
    if ($tRaw -match '^const\s+void\s*\*\*$') { return $null }
    # For pointer-to-pointer types (e.g. const char **), preserve const
    # to avoid char** -> const char** implicit conversion error in C++.
    if ($tRaw -match '\*\*\s*$') {
        $base = $tRaw -replace '\*\s*$', ''
        return "$($base.Trim()) local_$($p.Name){};"
    }
    $base = ($tRaw -replace '^\s*const\s+', '') -replace '\*\s*$', ''
    return "$($base.Trim()) local_$($p.Name){};"
}

function Is-DestroyFunction([string]$fname) {
    return $fname -match 'Destroy$'
}

function Get-DestroyClearCode([string]$fname, $parsed) {
    if (-not (Is-DestroyFunction $fname)) { return '' }
    if ($parsed.Count -eq 0) { return '' }
    $firstParam = $parsed[0]
    $varName = $firstParam.Name
    return "    $varName = {};"
}

function Get-NullGuardCode([string]$fname, $parsed) {
    foreach ($p in $parsed) {
        $tRaw = $p.Type.Trim()
        if ($tRaw -match '\*\s*$') { continue }
        $t = $tRaw -replace '^\s*const\s+', ''
        if ($t -eq 'UPTKfunction') { return @{ var = 'kern'; label = 'function' } }
        if ($t -eq 'UPTKmodule')   { return @{ var = 'mod'; label = 'module' } }
        if ($t -eq 'UPTKMemPool_t') { return @{ var = 'memPool'; label = 'mempool' } }
        if ($t -eq 'UPTKExternalMemory_t') { return @{ var = 'extMem'; label = 'external memory' } }
        if ($t -eq 'UPTKExternalSemaphore_t') { return @{ var = 'extSem'; label = 'external semaphore' } }
        if ($t -eq 'UPTKtexref')  { return @{ var = 'texRef'; label = 'texture reference' } }
        if ($t -eq 'UPTKlinkState') { return @{ var = 'linkState'; label = 'link state' } }
    }
    return $null
}

$rx = [regex]::Matches($text, '(?m)^UPTKError\s+(UP\w+)\(([^\)]*)\)\s*\{')
Write-Host "Matched $($rx.Count) wrappers"

$declPath = Join-Path $PSScriptRoot 'driver_smoke_decls.h'
$declSb = New-Object System.Text.StringBuilder
[void]$declSb.AppendLine('#pragma once')
[void]$declSb.AppendLine('/* Auto-generated from src/driver/driver_fun_convert.cpp - run generate_driver_tests.ps1 */')
[void]$declSb.AppendLine('#include <cuda.h>')
[void]$declSb.AppendLine('#include "driver_smoke_types.h"')
[void]$declSb.AppendLine('#include <UPTK.h>')
[void]$declSb.AppendLine('')
[void]$declSb.AppendLine('#ifdef __cplusplus')
[void]$declSb.AppendLine('extern "C" {')
[void]$declSb.AppendLine('#endif')
[void]$declSb.AppendLine('')
foreach ($dm in $rx) {
    $plDecl = $dm.Groups[2].Value.Trim()
    [void]$declSb.AppendLine(('UPTKError {0}({1});' -f $dm.Groups[1].Value, $plDecl))
}
[void]$declSb.AppendLine('')
[void]$declSb.AppendLine('#ifdef __cplusplus')
[void]$declSb.AppendLine('}')
[void]$declSb.AppendLine('#endif')
[void]$declSb.AppendLine('')
[System.IO.File]::WriteAllText($declPath, $declSb.ToString(), [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote driver_smoke_decls.h ($($rx.Count) prototypes)"

foreach ($m in $rx) {
    $fname = $m.Groups[1].Value
    $plist = $m.Groups[2].Value
    $parsed = @(
        foreach ($s in (Split-ParamList $plist)) {
            $x = Parse-ParamSegment $s
            if ($null -ne $x) { $x }
        }
    )

    $locals = New-Object System.Collections.Generic.List[string]
    foreach ($pp in $parsed) {
        $ld = LocalDecl-ForParam $pp
        if ($ld) { $locals.Add($ld) }
    }

    $args = foreach ($pp in $parsed) { Expr-ForParam $pp }
    $protoArgs = ($args -join ', ')

    $glev = Get-GraphLevel $fname $parsed
    $needEvt = Needs-Event $parsed
    $needArr = Needs-Array $parsed
    $needMip = Needs-Mipmap $parsed
    $needTex = Needs-TexRef $parsed

    $destroyClear = Get-DestroyClearCode $fname $parsed
    $nullGuard = Get-NullGuardCode $fname $parsed

    $localsJoined = (($locals | Sort-Object -Unique) -join "`n    ")

    $fixture = @"

static char dummy_pci_bus_id[] = "00000000:00:00.0";

static UPTKError driver_smoke_setup(
    UPTKdevice *dev,
    UPTKcontext *ctx,
    UPTKStream_t *stream,
    UPTKdeviceptr *dptr,
    UPTKEvent_t *evt,
    UPTKGraph_t *graph,
    UPTKGraphNode_t *graphNode,
    UPTKGraphExec_t *graphExec,
    UPTKarray *arr,
    UPTKMipmappedArray_t *mipmap,
    UPTKtexref *texRef,
    int graph_level,
    bool need_evt,
    bool need_arr,
    bool need_mipmap,
    bool need_tex)
{
    UPTKError err = UPInit(0);
    if (err != UPTKSuccess) return err;

    int ndev = 0;
    err = UPDeviceGetCount(&ndev);
    if (err != UPTKSuccess || ndev <= 0) return err;

    err = UPDeviceGet(dev, 0);
    if (err != UPTKSuccess) return err;

    err = UPDevicePrimaryCtxRetain(ctx, *dev);
    if (err != UPTKSuccess) return err;

    err = UPCtxSetCurrent(*ctx);
    if (err != UPTKSuccess) return err;

    err = UPStreamCreate(stream, 0u);
    if (err != UPTKSuccess) return err;

    err = UPMemAlloc_v2(dptr, 256);
    if (err != UPTKSuccess) return err;

    if (need_evt) {
        err = UPEventCreate(evt, 0u);
        if (err != UPTKSuccess) return err;
    }

    if (need_arr) {
        UPTK_ARRAY_DESCRIPTOR ad{};
        ad.Width = 64;
        ad.Height = 0;
        ad.Format = (UPTKarray_format)CU_AD_FORMAT_UNSIGNED_INT8;
        ad.NumChannels = 1;
        err = UPArrayCreate(arr, &ad);
        if (err != UPTKSuccess) return err;
    }

    if (need_mipmap) {
        UPTK_ARRAY3D_DESCRIPTOR td{};
        td.Width = 16;
        td.Height = 16;
        td.Depth = 1;
        td.Format = CU_AD_FORMAT_UNSIGNED_INT8;
        td.NumChannels = 1;
        td.Flags = 0;
        err = UPMipmappedArrayCreate(mipmap, &td, 1u);
        if (err != UPTKSuccess) return err;
    }

    if (graph_level >= 0) {
        err = UPGraphCreate(graph, 0u);
        if (err != UPTKSuccess) return err;
    }
    if (graph_level >= 1) {
        err = UPGraphAddEmptyNode(graphNode, *graph, nullptr, 0);
        if (err != UPTKSuccess) return err;
    }
    if (graph_level >= 2) {
        err = UPGraphInstantiateWithFlags(graphExec, *graph, 0ull);
        if (err != UPTKSuccess) return err;
    }

    if (need_tex && texRef) {
        err = UPTexRefCreate(texRef);
        if (err != UPTKSuccess) return err;
    }

    return UPTKSuccess;
}

static void driver_smoke_teardown(
    UPTKdevice dev,
    UPTKcontext ctx,
    UPTKStream_t stream,
    UPTKdeviceptr dptr,
    UPTKEvent_t evt,
    UPTKGraph_t graph,
    UPTKGraphExec_t graphExec,
    UPTKarray arr,
    UPTKMipmappedArray_t mipmap,
    UPTKtexref texRef,
    int graph_level,
    bool need_evt,
    bool need_arr,
    bool need_mipmap,
    bool need_tex)
{
    if (need_tex && texRef) {
        UPTexRefDestroy(texRef);
    }
    if (graph_level >= 2 && graphExec) {
        UPGraphExecDestroy(graphExec);
    }
    if (graph_level >= 0 && graph) {
        UPGraphDestroy(graph);
    }
    if (need_evt && evt) {
        UPEventDestroy(evt);
    }
    if (need_mipmap && mipmap) {
        UPMipmappedArrayDestroy(mipmap);
    }
    if (need_arr && arr) {
        UPArrayDestroy(arr);
    }
    if (dptr) {
        UPMemFree(dptr);
    }
    if (stream) {
        UPStreamDestroy(stream);
    }
    UPCtxSetCurrent((UPTKcontext)(uintptr_t)0);
}

"@

    $cpp = @"
/*
 * Auto-generated smoke test for driver wrapper $fname (driver_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/driver_test/generate_driver_tests.ps1
 */
#include <cuda.h>
#include "driver_smoke_types.h"
#include <UPTK.h>
#include "driver_smoke_decls.h"
#include <stdint.h>
#include <stdio.h>

$fixture

int main(void)
{
    UPTKdevice dev{};
    UPTKcontext ctx{};
    UPTKStream_t stream{};
    UPTKdeviceptr dptr{};
    UPTKEvent_t evt{};
    UPTKGraph_t graph{};
    UPTKGraphNode_t graphNode{};
    UPTKGraphExec_t graphExec{};
    UPTKarray arr{};
    UPTKMipmappedArray_t mipmap{};
    UPTKmodule mod{};
    UPTKfunction kern{};
    UPTKtexref texRef{};
    UPTKExternalMemory_t extMem{};
    UPTKExternalSemaphore_t extSem{};
    UPTKGraphicsResource_t gfxRes{};
    UPTKsurfref surfRef{};
    UPTKSurfaceObject_t surfObj{};
    UPTKTextureObject_t texObj{};
    UPTKMemPool_t memPool{};
    UPTKUserObject_t userObj{};
    UPTKlinkState linkState{};
    UPTKStreamCaptureStatus captureStatus{};
    $localsJoined

    const int graph_level = $glev;
    const bool need_evt = $(if ($needEvt) { 'true' } else { 'false' });
    const bool need_arr = $(if ($needArr) { 'true' } else { 'false' });
    const bool need_mipmap = $(if ($needMip) { 'true' } else { 'false' });
    const bool need_tex = $(if ($needTex) { 'true' } else { 'false' });

    UPTKError err = driver_smoke_setup(
        &dev,
        &ctx,
        &stream,
        &dptr,
        &evt,
        &graph,
        &graphNode,
        &graphExec,
        &arr,
        &mipmap,
        &texRef,
        graph_level,
        need_evt,
        need_arr,
        need_mipmap,
        need_tex);

    if (err != UPTKSuccess) {
        printf("test_skip: $fname setup failed (%d)\n", (int)err);
        return 0;
    }

$(if ($nullGuard) { @"
    if (!$($nullGuard.var)) {
        printf("test_skip: $fname needs valid $($nullGuard.label)\n");
    } else {
"@ } else { '' })
    err = $fname($protoArgs);$(if ($destroyClear) { "`n    $destroyClear" } else { '' })

$(if ($nullGuard) { @"
        printf("$fname -> %d\n", (int)err);
    }
"@ } else { @"
    printf("$fname -> %d\n", (int)err);
"@ })
    driver_smoke_teardown(
        dev,
        ctx,
        stream,
        dptr,
        evt,
        graph,
        graphExec,
        arr,
        mipmap,
        texRef,
        graph_level,
        need_evt,
        need_arr,
        need_mipmap,
        need_tex);

    printf("test_${fname} PASS\n");
    return 0;
}
"@

    $outPath = Join-Path $outDir ("test_{0}.cpp" -f $fname)
    Set-Content -LiteralPath $outPath -Value $cpp -Encoding UTF8
}

Write-Host "Wrote tests to $outDir"
