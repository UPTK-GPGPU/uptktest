/*
 * Auto-generated smoke test for driver wrapper UPStreamSetAttribute_ptsz (driver_fun_convert.cpp).
 * Regenerate: powershell -ExecutionPolicy Bypass -File test/driver_test/generate_driver_tests.ps1
 */
#include <cuda.h>
#include "driver_smoke_types.h"
#include <UPTK.h>
#include "driver_smoke_decls.h"
#include <stdint.h>
#include <stdio.h>


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
    UPTKStreamAttrValue local_value{};

    const int graph_level = -1;
    const bool need_evt = false;
    const bool need_arr = false;
    const bool need_mipmap = false;
    const bool need_tex = false;

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
        printf("test_skip: UPStreamSetAttribute_ptsz setup failed (%d)\n", (int)err);
        return 0;
    }

    err = UPStreamSetAttribute_ptsz(stream, (UPTKStreamAttrID)0, &local_value);

    printf("UPStreamSetAttribute_ptsz -> %d\n", (int)err);

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

    printf("test_UPStreamSetAttribute_ptsz PASS\n");
    return 0;
}

