/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#include <stdio.h>
#include <gtest/gtest.h>
#include <cuda.h>
//#include "test_common.h"
#include "MultiProcess.h"


#define NUM_ELEMS 1024
#define OFFSET 0 
static int g_num =0;
TEST(cudaMemory,cudaMultiProcIpcMem) {
#ifdef __unix__
  int num_process = 1;
  bool debug_process = false;
  int* ipc_dptr = nullptr;
  int* ipc_hptr = nullptr;
  int* ipc_out_dptr = nullptr;
  int* ipc_out_hptr = nullptr;
  int* ipc_offset_dptr = nullptr;
  UPTKError_t ret = UPTKSuccess;
  //int* ipc_dptr = nullptr;
  #if 0
  int* ipc_dptr_1 = nullptr;
  int * ipc_out_hptr_1 = new int[NUM_ELEMS];

  memset(ipc_out_hptr_1, 0x00, (NUM_ELEMS * sizeof(int)));
 
  ret = UPTKMalloc((void**)&ipc_dptr_1, NUM_ELEMS * sizeof(int));
  EXPECT_EQ(ret, UPTKSuccess);

  ret = UPTKMemcpy(ipc_dptr_1, ipc_out_hptr_1, (NUM_ELEMS * sizeof(int)), UPTKMemcpyHostToDevice);
  EXPECT_EQ(ret, UPTKSuccess);
  #endif
  MultiProcess<UPTKIpcMemHandle_t>* mProcess = new MultiProcess<UPTKIpcMemHandle_t>(num_process);
  printf("--- begin create shmem\n");
  mProcess->CreateShmem();
  printf("--- begin spawnprocess shmem\n");
  pid_t pid = mProcess->SpawnProcess(debug_process);
  printf("--- spawnprocess pid:%d\n",pid);

  // Parent Process
  if (pid != 0) {
    printf("--- begin parent procee:pid=%d\n", getpid());
    g_num = 1;
    printf("--------------------- begin parent procee:pid=%d,g_num=%d\n", getpid(),g_num);
    UPTKIpcMemHandle_t ipc_handle;
    memset(&ipc_handle, 0x00, sizeof(UPTKIpcMemHandle_t));
    ret = UPTKMalloc((void**)&ipc_dptr, NUM_ELEMS * sizeof(int));
    //ret =hipExtMallocWithFlags((void**)&ipc_dptr, NUM_ELEMS * sizeof(int),hipDeviceMallocFinegrained);
    EXPECT_EQ(ret, UPTKSuccess);
    printf(" hip malloc success,ipc_dptr:%p,line:%d\n",ipc_dptr,__LINE__);
    // Add offset to the dev_ptr
    ipc_offset_dptr = ipc_dptr + OFFSET;
    // Get handle for the offsetted device_ptr
    ret = UPTKIpcGetMemHandle(&ipc_handle, ipc_offset_dptr);
    EXPECT_EQ(ret, UPTKSuccess);
    printf(" UPTKIpcGetMemHandle success, line:%d\n",__LINE__);

    ipc_hptr = new int[NUM_ELEMS];
    for (size_t idx = 0; idx < NUM_ELEMS; ++idx) {
      ipc_hptr[idx] = idx;
    }

    ret = UPTKMemset(ipc_dptr, 0x00, (NUM_ELEMS * sizeof(int)));
    printf(" UPTKMemset success, line:%d\n",__LINE__);
    EXPECT_EQ(ret, UPTKSuccess);
    ret = UPTKMemcpy(ipc_dptr, ipc_hptr, (NUM_ELEMS * sizeof(int)), UPTKMemcpyHostToDevice);
    EXPECT_EQ(ret, UPTKSuccess);
    printf(" UPTKMemcpy  success, line:%d\n",__LINE__);

    mProcess->WriteHandleToShmem(ipc_handle);
    printf(" WriteHandleToShmem  line:%d\n",__LINE__);

    mProcess->WaitTillAllChildReads();
    printf(" WaitTillAllChildReads  line:%d\n",__LINE__);

  } else {
    printf("--- begin children procee:pid=%d\n", getpid());
    //ret = UPTKMemcpy(ipc_dptr_1, ipc_out_hptr_1, (NUM_ELEMS * sizeof(int)), UPTKMemcpyHostToDevice);
    //EXPECT_EQ(ret, UPTKSuccess);
    //g_num = 1;
    printf("--------------------- begin children procee:pid=%d,g_num=%d\n", getpid(),g_num);

    ipc_out_hptr = new int[NUM_ELEMS];
    memset(ipc_out_hptr, 0x00, (NUM_ELEMS * sizeof(int)));

    printf("children  ReadHandleFromShmem  line:%d\n",__LINE__);
    UPTKIpcMemHandle_t ipc_handle;
    mProcess->ReadHandleFromShmem(ipc_handle);
    printf(" ReadHandleFromShmem  line:%d\n",__LINE__);
    // Open handle to get dev_ptr
    ret = UPTKIpcOpenMemHandle((void**)&ipc_out_dptr, ipc_handle, UPTKDevAttrMaxThreadsPerBlock);
    EXPECT_EQ(ret, UPTKSuccess);
    printf("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| UPTKIpcOpenMemHandle ,ipc_out_dptr:%p line:%d\n",ipc_out_dptr,__LINE__);

    ret = UPTKMemcpy(ipc_out_hptr, ipc_out_dptr, (NUM_ELEMS * sizeof(int)),
                       UPTKMemcpyDeviceToHost);
    EXPECT_EQ(ret, UPTKSuccess);
    printf(" UPTKMemcpy  line:%d\n",__LINE__);
    for (size_t idx = 0; idx < NUM_ELEMS; ++idx) {
      if (ipc_out_hptr[idx] != idx) {
        std::cout<<"Failing @ idx: "<< idx << std::endl;
        EXPECT_EQ(ipc_out_hptr[idx], idx);
        break;
      }
    }
    mProcess->NotifyParentDone();
    printf(" NotifyParentDone  line:%d\n",__LINE__);
    ret = UPTKIpcCloseMemHandle(ipc_out_dptr);
    EXPECT_EQ(ret, UPTKSuccess);

    delete[] ipc_out_hptr;
  }

  if (pid != 0) {
    delete mProcess;
  }

#endif /* __unix__ */

}


