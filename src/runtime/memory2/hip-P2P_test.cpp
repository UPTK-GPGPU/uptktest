#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "cuda.h"


#define STREAM_NUM 2

#define READ_FLAG  1
#define WRITE_FLAG 2

#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

// Device (Kernel) function, it must be void
__global__ void p2PKernelTest_14(volatile unsigned long * pP2PMem, int flag) {

    volatile unsigned long* pMemAddress = pP2PMem;
    unsigned long  value = 0;
    if (READ_FLAG == flag )
    {
        __builtin_amdgcn_s_sleep(10000);
        //int loop=0;
        //while(loop<100){
        //    loop++;
        //}
        value = LOAD(pMemAddress);
        while (value < 100) {
            value = LOAD(pMemAddress);
        }
    } else if (WRITE_FLAG == flag) {
        #if 1 
        while (value < 100) {
            value++;
        }
        #endif
        for(int i =0;i<1000;i++){
          STORE(pMemAddress, 200);
           pMemAddress++;
        }
    }
}

// Device (Kernel) function, it must be void
__global__ void p2pkernelRead(volatile unsigned long * pP2PMem,  int flag) {

    volatile unsigned long* pMemAddress = pP2PMem;
    unsigned long  value = 0;
    if (READ_FLAG == flag )
    {
        //__builtin_amdgcn_s_sleep(10000);
        //int loop=0;
        //while(loop<100){
        //    loop++;
        //}
        value = LOAD(pMemAddress);
        #if 0
        while (value < 100) {
            value = LOAD(pMemAddress);
        }
        #endif
    } else if (WRITE_FLAG == flag) {
        #if 1 
        while (value < 100) {
            value++;
        }
        #endif
        for(int i =0;i<1000;i++){
          STORE(pMemAddress, 200);
           pMemAddress++;
        }
    }
}
#if 0
void getValue(volatile unsigned long * pP2PMem){
    unsigned long  value_old = 0xfffffff;
    unsigned long  value_new = 0xfffffff;
    unsigned long  * d_value = nullptr;
    d_value = UPTKMalloc(&d_value, sizeof(unsigned long));

    for(int i=0; i < 1000; i++){
        //hipLaunchKernelGGL(p2pkernelRead, 1, 1, 0, 0, pP2PMem, d_value);
        p2pkernelRead<<<1, 1>>> (pP2PMem, d_value);
        UPTKMemcpy(value_new, d_value, sizeof(unsigned long), UPTKMemcpyDeviceToHost);

        if(value_new != value_old){
            value_old = value_new;
            printf("value: %ld\n", value_new);
        }
    }

    UPTKFree(d_value);
}
#endif
UPTKStream_t stream3 = NULL;
unsigned long * pDevice = NULL;

void* ReadStream(void* data){
    UPTKError_t ret = UPTKSuccess;
    printf("----------------------Read Stream-----------------\n");
    
    ret =  UPTKSetDevice(0);
    if (UPTKSuccess != ret){
           printf("Error: hip set device :%d failed\n",0);
           return NULL;
    }
    for (int i=0;i<1;i++) {
    //hipLaunchKernelGGL(p2pkernelRead, 
    //        dim3(1),
    //        dim3(1),
    //        0,
    //        stream3, 
    //        pDevice,
    //        READ_FLAG); 
    p2pkernelRead<<<dim3(1),
            dim3(1),
            0,
            stream3>>>(
            pDevice,
            READ_FLAG);
    printf("\t ************************* wait stream3: %p ************************\n", stream3);
    }
    #if 1 
    ret = UPTKStreamSynchronize(stream3);
    if (UPTKSuccess != ret){
      printf("Error cuda set device :%d failed\n",ret);
      return NULL;
    }
    #endif    
    printf("\t ************************** finish wait stream3: %p-----------------------\n", stream3);
    return NULL;
}

int main() {
    UPTKStream_t strem[STREAM_NUM];
    UPTKError_t ret = UPTKSuccess;
    int ngpus;
    unsigned long   host = 50;
    //unsigned long * pDevice = NULL;

    //UPTKStream_t stream3 = NULL;
    
    // check device count
    ret = UPTKGetDeviceCount(&ngpus);
    if (UPTKSuccess != ret){
        printf("Error: get device count failed\n");
        return ret;
    }

    if (ngpus < STREAM_NUM) {
        printf("Error:there are not enough gpu cards, the gpu cards are at least:%d\n",STREAM_NUM);
    }
    ret =  UPTKSetDevice(0);
    if (UPTKSuccess != ret){
        printf("Error: cuda set device :%d failed\n",0);
        return ret;
    }
    printf("----- UPTKMalloc ----\n");     
    ret = hipExtMallocWithFlags((void **)&pDevice, sizeof(unsigned long)*1024,hipDeviceMallocFinegrained);    
    if (UPTKSuccess != ret){
        printf("Error: hip malloc failed,ret=%d\n",ret);
        return ret;
    }
    printf("\t hip malloc address:%p\n",pDevice);

    int canAccessPeer = 0;
    ret = UPTKDeviceCanAccessPeer(&canAccessPeer, 1, 0);
    if (UPTKSuccess != ret){
        printf("Error: hip malloc failed\n");
        return ret;
    }
    if (!canAccessPeer) {
         printf("Error: P2P can not access peer\n");
         return 0;
    }
    //*pDevice = host;
    *pDevice = 50;
    printf("----------------------------------------------------value:%lu\n",*pDevice);
    #if 1 
    ret = UPTKMemcpy(pDevice, &host, sizeof(unsigned long),  UPTKMemcpyHostToDevice);
    if (UPTKSuccess != ret){
        printf("Error: hip memory failed\n");
        return ret;
    }
    #endif

    ret =  UPTKSetDevice(1);
    if (UPTKSuccess != ret){
        printf("Error: hip set device :%d failed\n",0);
        return ret;
    }
    #if 0 
    ret = UPTKMemcpy(pDevice, &host, sizeof(unsigned long),  UPTKMemcpyHostToDevice);
    if (UPTKSuccess != ret){
        printf("Error: hip memory failed\n");
        return ret;
    }
    #endif
    ret = UPTKDeviceEnablePeerAccess(0, 0);
    if (UPTKSuccess != ret){
        printf("Error: hip memory failed\n");
        return ret;
    }
        
    for (int i=0;i<STREAM_NUM;i++) {
        ret =  UPTKSetDevice(i);
        if (UPTKSuccess != ret){
            printf("Error: hip set device :%d failed\n",i);
            return ret;
        }
        ret = UPTKStreamCreate(&strem[i]);
        if (UPTKSuccess != ret){
            printf("Error: hip create stream,on device :%d failed\n",i);
            return ret;
        }
    }
    #if 1 
    ret =  UPTKSetDevice(0);
    if (UPTKSuccess != ret){
        printf("Error: hip set device :%d failed\n",0);
        return ret;
    }
    //hipLaunchKernelGGL( p2PKernelTest_14, 
    //                dim3(1),
    //                dim3(1),
    //                0,
    //                strem[0], 
    //                pDevice,
    //                READ_FLAG); 
    p2PKernelTest_14<<<dim3(1),
                    dim3(1),
                    0,
                    strem[0]>>>(
                    pDevice,
                    READ_FLAG);
    #endif
    sleep(2);
    ret =  UPTKSetDevice(1);
    if (UPTKSuccess != ret){
        printf("Error: hip set device :%d failed\n",0);
        return ret;
    }
   
    //hipLaunchKernelGGL(p2PKernelTest_14, 
    //               dim3(1),
     //               dim3(1),
     //               0,
     //               strem[1], 
     //               pDevice,
     //               WRITE_FLAG); 
    p2PKernelTest_14<<< dim3(1),
                    dim3(1),
                    0,
                    strem[1]>>>( 
                    pDevice,
                    WRITE_FLAG);
    #if 1 
    sleep(2);
    ret =  UPTKSetDevice(0);
      #if 0
    ret = UPTKStreamCreate(&stream3);
    if (UPTKSuccess != ret){
        printf("Error: hip create stream,on device :%d failed\n",ret);
        return ret;
    }
      #endif
    #if 0 
    if (UPTKSuccess != ret){
        printf("Error: hip set device :%d failed\n",0);
        return ret;
    }
    //hipLaunchKernelGGL( p2PKernelTest_14, 
    //                dim3(1),
    //                dim3(1),
    //                0,
    //                strem[0], 
    //                pDevice,
    //                READ_FLAG);
    p2PKernelTest_14<<< dim3(1),
                    dim3(1),
                    0,
                    strem[0]>>>( 
                    pDevice,
                    READ_FLAG);
    #endif 
    #endif
    printf("------ all the kernel has launched\n");

    //pthread_t thread1;
    //pthread_create(&thread1, 0, ReadStream, 0);
    
    //for (int i=0;i<STREAM_NUM;i++) {
    for (int i=(STREAM_NUM-1);i>=0;i--) {
        printf("\t begin sync stream:%d\n",i);
        ret =  UPTKSetDevice(i);
        if (UPTKSuccess != ret){
            printf("Error: hip set device :%d failed\n",i);
            return ret;
        }

        ret = UPTKStreamSynchronize(strem[i]);
        if (UPTKSuccess != ret){
            printf("Error: hip strem  syn device :%d failed\n",i);
            return ret;
        }
         printf("\t finish stream:%d\n",i);
    }

    //pthread_join(thread1,NULL);
    UPTKSetDevice(0);
    printf("----------------------------------------------------value:%lu\n",*pDevice);
    UPTKFree(pDevice);
    
    for (int i=0;i<STREAM_NUM;i++) {
          ret =  UPTKSetDevice(i);
          if (UPTKSuccess != ret){
              printf("Error: hip set device :%d failed\n",i);
              return ret;
          }
    
          ret = UPTKStreamDestroy(strem[i]);
          if (UPTKSuccess != ret){
              printf("Error: hip set device :%d failed\n",i);
              return ret;
          }
      }
    printf("success-----\n");
}
