#pragma once

/*
 * Driver wrapper layer uses CU and CUDA struct typedef aliases (see src/driver/driver.hpp).
 * Public UPTK.h uses different spellings for some types (e.g. UPTKFuncCache vs UPTKfunc_cache).
 * This header mirrors driver.hpp so driver_smoke_decls.h prototypes compile with only
 * UPTK.h + CUDA headers pulled by tests.
 */

/* Define OpenGL types for UPGraphicsGL* prototypes before including cuda.h */
#ifndef GLuint
typedef unsigned int GLuint;
#endif
#ifndef GLenum
typedef unsigned int GLenum;
#endif
#ifndef CUGLDeviceList
typedef unsigned int CUGLDeviceList;
#endif

#include <cuda.h>

#ifndef CU_AD_FORMAT_UNSIGNED_INT8
#define CU_AD_FORMAT_UNSIGNED_INT8 ((CUarray_format)0x01)
#endif

#ifndef UPTK_DRIVER_SMOKE_LEGACY_CUDA_TYPEDEFS
#define UPTK_DRIVER_SMOKE_LEGACY_CUDA_TYPEDEFS

typedef CUaddress_mode                              UPTKaddress_mode;
#ifdef CUarrayMapInfo
typedef CUarrayMapInfo                              UPTKarrayMapInfo;
#else
typedef void*                                       UPTKarrayMapInfo;
#endif
#ifdef CUdevice_P2PAttribute
typedef CUdevice_P2PAttribute                       UPTKdevice_P2PAttribute;
#else
typedef unsigned int                                UPTKdevice_P2PAttribute;
#endif
typedef CUdevprop                                   UPTKdevprop;
#ifdef CUexecAffinityParam
typedef CUexecAffinityParam                         UPTKexecAffinityParam;
#else
typedef void*                                       UPTKexecAffinityParam;
#endif
#ifdef CUexecAffinityType
typedef CUexecAffinityType                          UPTKexecAffinityType;
#else
typedef unsigned int                                UPTKexecAffinityType;
#endif
typedef CUfilter_mode                               UPTKfilter_mode;
typedef CUfunc_cache                                UPTKfunc_cache;
typedef CUGLDeviceList                              UPTKGLDeviceList;
#ifdef CUgraphMem_attribute
typedef CUgraphMem_attribute                        UPTKgraphMem_attribute;
#else
typedef unsigned int                                UPTKgraphMem_attribute;
#endif
typedef CUhostFn                                    UPTKhostFn;
typedef CUmem_advise                                UPTKmem_advise;
#ifdef CUmem_range_attribute
typedef CUmem_range_attribute                       UPTKmem_range_attribute;
#else
typedef unsigned int                                UPTKmem_range_attribute;
#endif
#ifdef CUmemAccess_flags
typedef CUmemAccess_flags                           UPTKmemAccess_flags;
#else
typedef unsigned int                                UPTKmemAccess_flags;
#endif
#ifdef CUmemAllocationGranularity_flags
typedef CUmemAllocationGranularity_flags            UPTKmemAllocationGranularity_flags;
#else
typedef unsigned int                                UPTKmemAllocationGranularity_flags;
#endif
#ifdef CUmemAllocationProp
typedef CUmemAllocationProp                         UPTKmemAllocationProp;
#else
typedef void*                                       UPTKmemAllocationProp;
#endif
#ifdef CUmemGenericAllocationHandle
typedef CUmemGenericAllocationHandle                UPTKmemGenericAllocationHandle;
#else
typedef unsigned long long                          UPTKmemGenericAllocationHandle;
#endif
typedef CUmemPool_attribute                         UPTKmemPool_attribute;
#ifdef CUmemRangeHandleType
typedef CUmemRangeHandleType                        UPTKmemRangeHandleType;
#else
typedef unsigned int                                UPTKmemRangeHandleType;
#endif
#ifdef CUmoduleLoadingMode
typedef CUmoduleLoadingMode                         UPTKmoduleLoadingMode;
#else
typedef unsigned int                                UPTKmoduleLoadingMode;
#endif
#ifdef CUoccupancyB2DSize
typedef CUoccupancyB2DSize                          UPTKoccupancyB2DSize;
#else
typedef size_t (*UPTKoccupancyB2DSize)(unsigned int);
#endif
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
typedef CUoutput_mode                               UPTKoutput_mode;
#else
typedef unsigned int                                UPTKoutput_mode;
#endif
typedef CUpointer_attribute                         UPTKpointer_attribute;
typedef CUsharedconfig                              UPTKsharedconfig;
typedef CUstreamBatchMemOpParams                    UPTKstreamBatchMemOpParams;
typedef CUstreamCallback                            UPTKstreamCallback;
typedef CUsurfref                                   UPTKsurfref;
typedef CUsurfObject                                UPTKSurfaceObject_t;
typedef CUtexObject                                 UPTKTextureObject_t;
typedef CUtexref                                    UPTKtexref;

typedef CUDA_ARRAY_MEMORY_REQUIREMENTS              UPTK_ARRAY_MEMORY_REQUIREMENTS;
#ifdef CUDA_ARRAY_SPARSE_PROPERTIES
typedef CUDA_ARRAY_SPARSE_PROPERTIES                UPTK_ARRAY_SPARSE_PROPERTIES;
#else
typedef void*                                       UPTK_ARRAY_SPARSE_PROPERTIES;
#endif
typedef CUDA_ARRAY3D_DESCRIPTOR                     UPTK_ARRAY3D_DESCRIPTOR;
#ifdef CUDA_BATCH_MEM_OP_NODE_PARAMS
typedef CUDA_BATCH_MEM_OP_NODE_PARAMS               UPTK_BATCH_MEM_OP_NODE_PARAMS;
#else
typedef void*                                       UPTK_BATCH_MEM_OP_NODE_PARAMS;
#endif
#ifdef CUDA_EXT_SEM_SIGNAL_NODE_PARAMS
typedef CUDA_EXT_SEM_SIGNAL_NODE_PARAMS             UPTK_EXT_SEM_SIGNAL_NODE_PARAMS;
#else
typedef void*                                       UPTK_EXT_SEM_SIGNAL_NODE_PARAMS;
#endif
#ifdef CUDA_EXT_SEM_WAIT_NODE_PARAMS
typedef CUDA_EXT_SEM_WAIT_NODE_PARAMS               UPTK_EXT_SEM_WAIT_NODE_PARAMS;
#else
typedef void*                                       UPTK_EXT_SEM_WAIT_NODE_PARAMS;
#endif
typedef CUDA_EXTERNAL_MEMORY_BUFFER_DESC            UPTK_EXTERNAL_MEMORY_BUFFER_DESC;
typedef CUDA_EXTERNAL_MEMORY_HANDLE_DESC            UPTK_EXTERNAL_MEMORY_HANDLE_DESC;
typedef CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC   UPTK_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC;
typedef CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC         UPTK_EXTERNAL_SEMAPHORE_HANDLE_DESC;
typedef CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS       UPTK_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS;
typedef CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS         UPTK_EXTERNAL_SEMAPHORE_WAIT_PARAMS;
typedef CUDA_HOST_NODE_PARAMS                       UPTK_HOST_NODE_PARAMS;
typedef CUDA_KERNEL_NODE_PARAMS                     UPTK_KERNEL_NODE_PARAMS;
typedef CUDA_LAUNCH_PARAMS                          UPTK_LAUNCH_PARAMS;
typedef CUDA_MEM_ALLOC_NODE_PARAMS                  UPTK_MEM_ALLOC_NODE_PARAMS;
typedef CUDA_MEMCPY2D                               UPTK_MEMCPY2D;
typedef CUDA_MEMCPY3D                               UPTK_MEMCPY3D;
typedef CUDA_MEMCPY3D_PEER                          UPTK_MEMCPY3D_PEER;
typedef CUDA_MEMSET_NODE_PARAMS                     UPTK_MEMSET_NODE_PARAMS;
typedef CUDA_RESOURCE_DESC                          UPTK_RESOURCE_DESC;
typedef CUDA_RESOURCE_VIEW_DESC                     UPTK_RESOURCE_VIEW_DESC;
typedef CUDA_TEXTURE_DESC                           UPTK_TEXTURE_DESC;

#endif /* UPTK_DRIVER_SMOKE_LEGACY_CUDA_TYPEDEFS */

/*
 * CUDA 12+ only (see driver.hpp). Older toolkits: opaque placeholders for declarations.
 */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
typedef CUasyncCallback                             UPTKAsyncCallback;
typedef CUasyncCallbackHandle                       UPTKasyncCallbackHandle;
typedef CUcoredumpSettings                          UPTKcoredumpSettings;
typedef CUctxCreateParams                           UPTKctxCreateParams;
typedef CUdevResource                               UPTKdevResource;
typedef CUdevResourceDesc                           UPTKdevResourceDesc;
typedef CUdevResourceType                           UPTKdevResourceType;
typedef CUdriverProcAddressQueryResult              UPTKdriverProcAddressQueryResult;
typedef CUfunctionLoadingState                      UPTKfunctionLoadingState;
typedef CUgraphEdgeData                             UPTKGraphEdgeData;
typedef CUgraphExecUpdateResultInfo                 UPTKGraphExecUpdateResultInfo;
typedef CUDA_GRAPH_INSTANTIATE_PARAMS               UPTK_GRAPH_INSTANTIATE_PARAMS;
typedef CUgreenCtx                                  UPTKgreenCtx;
typedef CUkernel                                    UPTKKernel_t;
typedef CUlibrary                                   UPTKlibrary;
typedef CUlibraryOption                             UPTKlibraryOption;
typedef CUmulticastGranularity_flags                UPTKmulticastGranularity_flags;
typedef CUmulticastObjectProp                       UPTKmulticastObjectProp;
typedef CUtensorMap                                 UPTKtensorMap;
typedef CUtensorMapDataType                         UPTKtensorMapDataType;
typedef CUtensorMapFloatOOBfill                     UPTKtensorMapFloatOOBfill;
typedef CUtensorMapInterleave                       UPTKtensorMapInterleave;
typedef CUtensorMapL2promotion                      UPTKtensorMapL2promotion;
typedef CUtensorMapSwizzle                          UPTKtensorMapSwizzle;
#else
typedef struct UPTKAsyncCallback_st *UPTKAsyncCallback;
typedef struct UPTKasyncCallbackHandle_st *UPTKasyncCallbackHandle;
typedef unsigned int UPTKcoredumpSettings;
typedef struct UPTKctxCreateParams_st {
    unsigned char _opaque[512];
} UPTKctxCreateParams;
typedef struct UPTKdevResource_st {
    unsigned char _opaque[512];
} UPTKdevResource;
typedef struct UPTKdevResourceDesc_st {
    unsigned char _opaque[512];
} UPTKdevResourceDesc;
typedef unsigned int UPTKdevResourceType;
typedef unsigned int UPTKdriverProcAddressQueryResult;
typedef unsigned int UPTKfunctionLoadingState;
typedef struct UPTKGraphEdgeData_st {
    unsigned char _opaque[256];
} UPTKGraphEdgeData;
typedef struct UPTKGraphExecUpdateResultInfo_st {
    unsigned char _opaque[256];
} UPTKGraphExecUpdateResultInfo;
typedef struct UPTK_GRAPH_INSTANTIATE_PARAMS_st {
    unsigned char _opaque[512];
} UPTK_GRAPH_INSTANTIATE_PARAMS;
typedef struct UPTKgreenCtx_st *UPTKgreenCtx;
typedef struct UPTKKernel_st *UPTKKernel_t;
typedef struct UPTKlibrary_st *UPTKlibrary;
typedef unsigned int UPTKlibraryOption;
typedef unsigned int UPTKmulticastGranularity_flags;
typedef struct UPTKmulticastObjectProp_st {
    unsigned char _opaque[256];
} UPTKmulticastObjectProp;
typedef struct UPTKtensorMap_st {
    unsigned char _opaque[256];
} UPTKtensorMap;
typedef unsigned int UPTKtensorMapDataType;
typedef unsigned int UPTKtensorMapFloatOOBfill;
typedef unsigned int UPTKtensorMapInterleave;
typedef unsigned int UPTKtensorMapL2promotion;
typedef unsigned int UPTKtensorMapSwizzle;
#endif
