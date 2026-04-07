/*
@BUG #36196 After using UPTKMemset, the size of set is inconsistent with the passed parameter size - DCUToolkit - Runtime
@Add Case For HIP-TEST
    Async = true
*/
#include <stdio.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>
#include "gtest/test_common.h"
TEST(cudaMemory,cudaMemsetTest1){
    char *mem;
    char *check_mem;
    char res;
    int test_rows = 256;
    int test_range = test_rows * 32;
    bool testAsync = true;
    CUDACHECK(UPTKMallocManaged((void **)&mem, test_range));
    CUDACHECK(UPTKMallocHost((void **)&check_mem, test_range));
    int set_size = 1;
    int offset = 1;
    while(true){
        if(testAsync) {
            CUDACHECK(UPTKMemsetAsync((void *)(mem + offset), 1, set_size));
        }
        else {
            CUDACHECK(UPTKMemset((void *)(mem + offset), 1, set_size));
        }
        offset += set_size + 1;
        set_size++;
        if(offset + set_size > test_range) {
            break;
        }
    }
    if(testAsync) {
       CUDACHECK(UPTKStreamSynchronize(0));
    }
    CUDACHECK(UPTKMemcpy(check_mem, mem, test_range, UPTKMemcpyDeviceToHost));
    CUDACHECK(UPTKMemcpy(check_mem, mem, test_range, UPTKMemcpyDeviceToHost));
    CUDACHECK(UPTKFree(mem));
    res = CudaTest::check_res(check_mem,test_range);
    ASSERT_EQ(true,res);
}