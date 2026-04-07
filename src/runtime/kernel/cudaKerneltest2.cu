#include <iostream>
#include <pthread.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <UPTK_runtime_api.h>
#include <UPTK_runtime.h>

namespace{
#define  Batch 100000
#define  M 1024
#define  N 1024
//#define  Batch 1

#define HIP_CHECK(stat)                                                        \
    ({                                                                          \
        UPTKError_t restat = stat;                                               \
        if(restat != UPTKSuccess)                                                 \
        {                                                                      \
            printf("Error: hip error in line %d,stat:%d\n",__LINE__,restat) ;      \
        }                                                                      \
    })


__global__ void set_C_to_1024(float *C_value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        C_value[tid] = 1024.0;
    }
}

__global__ void set_C_to_1(float *C_value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        C_value[tid] = 1.0;
    }
}

__global__ void set_B_to_124(float *C_value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        C_value[tid] = 124.0;
    }
}
__global__ void set_B_to_256(float *C_value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        C_value[tid] = 256.0;
    }
}
__global__ void set_C_copy_to_B(float *C_value, float *B_value, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        B_value[tid] = C_value[tid];
    }
}

int check_value(float *value,float *device, int n, int expect, std::string &&name) {
    int flag = 1;
    unsigned int error = 0;
    for (int i = 0; i < n; i++) {
        if (value[i] != expect)
            //std::cout<<name<<" value["<<i<<"]!="<<expect<<std::endl;
        {
            if (!error) {
                printf("\t%s value[%d]=%f != %d,device=%f thread_id=%u\n",name.c_str(),i,value[i],expect,device[i],(unsigned int)pthread_self());
            }
            error++;
            flag = 0;
        }
    }
    if (error) {
        printf("%s error number=%d thread_id=%u\n", name.c_str(), error, (unsigned int) pthread_self());
    }
    return flag;
    //std::cout<<name<<" check passed and expect is "<<expect<<std::endl;
}

void *csr_mm(void *data) {
    int flag_thread = 1;
    UPTKStream_t StreamData;
    UPTKError_t ret = UPTKSuccess;
    int id = *(int *) data;
    float time_use = 0;
    struct timeval start;
    struct timeval end;

    printf("--- device id:%d\n", id);

    ret = UPTKStreamCreate(&StreamData);
    if (ret) {
        printf("create stream error:%d\n", ret);
        return NULL;
    }

    for (int i = 0; i < Batch; i++) {
        if (!(i%1000)){
            printf("index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        }
        //std::cout<<"PID:"<<pthread_self()<<" is running."<<std::endl;
        //sparse MM A on host
        float *SMM_value;
        //    printf("22222 index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        HIP_CHECK(UPTKMallocHost((void **) &SMM_value, sizeof(float) * M * N));
        //SMM_value =(float *)malloc(sizeof(float) * M * N);
        //dense MM B on host
        float *den_value_b;
        //    printf("33333  index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        HIP_CHECK(UPTKMallocHost((void**)&den_value_b,sizeof(float)*M*N));
        //den_value_b  =(float *)malloc(sizeof(float) * M * N);
//        CreateDM(den_value_b,1.0f);

        //dense MM C on host
        float *den_value_c;
        HIP_CHECK(UPTKMallocHost((void**)&den_value_c,sizeof(float)*M*N));
        //    printf("11111 begin index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        //den_value_c  =(float *)malloc(sizeof(float) * M * N);
//        CreateDM(den_value_c,0.0f);

        float alpha = 1.0f;
        float beta = 0.0f;

        //device A
        float *d_A_value;
        HIP_CHECK(UPTKMalloc((void **) &d_A_value, sizeof(float) * M * N));
        //    printf("begin index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        HIP_CHECK(UPTKMemcpyAsync(d_A_value, SMM_value, sizeof(float) * M * N, UPTKMemcpyHostToDevice,StreamData));

        //device B
        float *d_B_value;
        HIP_CHECK(UPTKMalloc((void **) &d_B_value, sizeof(float) * M * N));

        //device C
        float *d_C_value;
        HIP_CHECK(UPTKMalloc((void **) &d_C_value, sizeof(float) * M * N));
       //printf("memory size A:%zu(M),B:%zu(M),C:%zu(M)\n",sizeof(float) * M * N/(1024*1024),sizeof(float) * M * N/(1024*1024),sizeof(float) * M * N/(1024*1024));
    //for (int i = 0; i < Batch; i++) {
    //    if (!(i%1000)){
    //        printf("index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
     //   }
        //设置流，不同线程用不同流
        UPTKStream_t *stream = (UPTKstream *)&StreamData;
        UPTKStream_t temp = *stream;

        //printf("first index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        //dim3 blocksize(256,1);
        dim3 blocksize(128, 1);
        dim3 gridsize(M * N / 128 , 1);
        //printf("kernel first thread_id=%u\n",(unsigned int)pthread_self());
        //hipLaunchKernelGGL(set_C_to_1024, gridsize, blocksize, 0, temp , d_C_value, M * N);
        set_C_to_1024<<<gridsize, blocksize, 0, temp>>> (d_C_value, M * N);

        UPTKStreamSynchronize(temp);
        //usleep(500 * 1);
        #if 0 
        if (1024 != d_C_value[M*N/2]){
            printf("-----error:after stream:%p sync first kernel d_C_value:%f, expect 1024,index:%d,line:%d\n",temp, d_C_value[M*N/2],i,__LINE__);
            #if 0
            for(int j=0;j<5; j++){
               printf("\t-----error:after stream:%p sync first kernel d_C_value[%d]:%f, expect 1024,index:%d\n",temp,(j*800), d_C_value[(j*800)],i);
            }
            #endif
            abort();
        }
        #endif
        //gettimeofday(&end,NULL);
        //time_use=(end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
        //printf("time_use is %.10f\n",time_use);
        //usleep(1000 * 500);
        //计算完毕检查是否是1024
        HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
        //printf("end kernel first thread_id=%u\n",(unsigned int)pthread_self());
        flag_thread = check_value(den_value_c,d_C_value, M * N, 1024, "C matrix");
        if (!flag_thread) {
            printf("first checkt error  index:%d,stream:%p,den_value_c=%p,d_C_value=%p,thread_id=%u\n",
                                  i,temp,den_value_c,d_C_value,(unsigned int)pthread_self());
            UPTKStreamQuery(temp);
            //HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
            unsigned first=0;
            for (int j = 0; j < M * N; j++)
            {
                if (den_value_c[j] != 1024) {
                    printf("den_value_c[%d]=%f d_C_value[%d]=%f thread_id=%u\n", j, den_value_c[j], j, d_C_value[j],
                           (unsigned int) pthread_self());
                    //break;
                    first++;
                    break;
                }
            }
            if (first){
                usleep(1000);
                abort();
                //HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost,temp));
                return NULL;
            }
        }
        //设置C值为1,然后检查
        //printf("kernel second thread_id=%u\n",(unsigned int)pthread_self());
        //hipLaunchKernelGGL(set_C_to_1, gridsize, blocksize, 0, temp , d_C_value, M * N);
        set_C_to_1<<<gridsize, blocksize, 0, temp>>> (d_C_value, M * N);
        //hipLaunchKernelGGL(set_C_to_1, 1, 256, 0, temp, d_C_value,M*N);
        ret = UPTKGetLastError();
        if (ret) {
            printf("Error kernel launch :%d\n", ret);
            return NULL;
        }
        //usleep(1000 * 50);
        UPTKStreamSynchronize(temp);
        //usleep(500 * 1);
        #if 0 
        if (1 != d_C_value[M*N/2]){
            printf("------error:after sync stream:%p second kernel d_C_value:%f,expect 1,index:%d,line:%d\n",temp, d_C_value[M*N/2],i,__LINE__);
            #if 0
            for(int j=0;j<5;j++){
                printf("\t------error:after sync stream:%p second kernel d_C_value[%d]:%f,expect 1,index:%d\n",temp,(j*800), d_C_value[j*800],i);
            }
            #endif 
            abort();
        }
        #endif
        //usleep(1000 * 50);
        //HIP_CHECK(UPTKMemcpy(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost));
        HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost,temp));
        //printf("end kernel second thread_id=%u\n",(unsigned int)pthread_self());
        flag_thread = check_value(den_value_c,d_C_value, M * N, 1, "C matrix");
        //flag_thread=check_value(den_value_c,256,1,"C matrix");
        //flag_thread=check_value(d_C_value,M*N,1,"C matrix");
        if (!flag_thread) {
            printf("second checkt error index:%d, stream=%p,den_value_c=%p,d_C_value=%p,thread_id=%u\n",
                           i,temp,den_value_c,d_C_value, (unsigned int) pthread_self());
            UPTKStreamQuery(temp);
#if 0
            //hipLaunchKernelGGL(set_C_to_1, gridsize, blocksize, 0, temp, d_C_value,M*N);
            set_C_to_1<<<gridsize, blocksize, 0, temp>>?(d_C_value,M*N);
            HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost,temp));
            flag_thread=check_value(den_value_c,M*N,1,"C matrix");
            //flag_thread=check_value(d_C_value,M*N,1,"C matrix");
            if (!flag_thread) {
                printf("second checkt error again index:%d, thread_id=%u\n",i,(unsigned int)pthread_self());
            }
#endif
#if 1
            //HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_C_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost,temp));
            unsigned int compare = 0;
            for (int j = 0; j < M * N; j++)
                //for (int j=0;j<128;j++)
            {
                if (den_value_c[j] != 1) {
                    printf("den_value_c[%d]=%f d_C_value[%d]=%f thread_id=%u\n", j, den_value_c[j], j, d_C_value[j],
                           (unsigned int) pthread_self());
                    compare++;
                    break;
                }
            }
#endif
            if (compare){
                usleep(1000);
                abort();
                return NULL;
            }
        }
        //printf("kernel third thread_id=%u\n",(unsigned int)pthread_self());
        //C赋值给B,然后检查B值是否为1
        //hipLaunchKernelGGL(set_C_copy_to_B, gridsize, blocksize, 0, temp , d_C_value, d_B_value, M * N);
        set_C_copy_to_B<<<gridsize, blocksize, 0, temp>>>(d_C_value, d_B_value, M * N);

        //hipLaunchKernelGGL(set_C_copy_to_B, 1, 256, 0, temp, d_C_value,d_B_value,M*N);
        //usleep(1000 * 500);
        UPTKStreamSynchronize(temp);
        //usleep(500 * 1);
        #if 0 
        if (1 != d_B_value[M*N/2]){
            printf("-----error -after sync stream:%p  second kernel d_B_value:%f, expect 1, index:%d,line:%d\n",temp, d_B_value[M*N/2],i,__LINE__);
            #if 0
            for(int j=0;j<5; j++){
                printf("\t-----error -after sync stream:%p  second kernel d_B_value[%d]:%f, expect 1, index:%d\n",temp,(j*800),d_B_value[(j*800)],i);
            }
            #endif
            abort();
        }
        #endif
        HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_B_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
        //printf("end kernel third thread_id=%u\n",(unsigned int)pthread_self());
        flag_thread = check_value(den_value_c,d_B_value, M * N, 1, "B matrix");
        //flag_thread=check_value(den_value_c,d_B_value,256,1,"B matrix");
        if (!flag_thread) {
            printf("third checkt error index:%d, stream:%p,den_value_c=%p,d_B_value=%p,thread_id=%u\n",
                             i,temp,den_value_c,d_B_value, (unsigned int) pthread_self());
            UPTKStreamQuery(temp);
            //HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_B_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
            unsigned int forth=0;            
            for (int j = 0; j < M * N; j++)
            {
                if (den_value_c[j] != 1) {
                    printf("den_value_c[%d]=%f d_B_value[%d]=%f thread_id=%u\n", j, den_value_c[j], j, d_B_value[j],
                           (unsigned int) pthread_self());
                    forth++;
                    break;
                }
            }
            if (forth) {
                usleep(1000);
                abort();
                return NULL;
            }
        }
        #if 1 
        //printf("kernel forth thread_id=%u\n",(unsigned int)pthread_self());
        //hipLaunchKernelGGL(set_B_to_124, gridsize, blocksize, 0, temp , d_B_value, M * N);
        set_B_to_124<<<gridsize, blocksize, 0, temp>>> (d_B_value, M * N);

        UPTKStreamSynchronize(temp);
        //usleep(500 * 1);
        #if 0 
        if (124 != d_B_value[M*N/2]){
            printf("----error:after sync stream:%p forth  kernel d_B_value:%f, expect 124, index:%d,line:%d\n",temp ,d_B_value[M*N/2],i,__LINE__);
            #if 0
            for(int j=0;j<5; j++){
                printf("\t----error:after sync stream:%p forth  kernel d_B_value[%d]:%f, expect 124, index:%d\n",temp ,(j*800),d_B_value[(j*800)],i);
            }
            #endif
            abort();
        }
        #endif
        HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_B_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
        //printf("end kernel forth thread_id=%u\n",(unsigned int)pthread_self());
        flag_thread = check_value(den_value_c,d_B_value, M * N, 124, "B matrix");
        if (!flag_thread) {
            printf("forth checkt error index:%d,stream:%p,den_value_c=%p,d_B_value=%p thread_id=%u\n",
                                 i,temp,den_value_c,d_B_value, (unsigned int) pthread_self());
            UPTKStreamQuery(temp);
            //HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_B_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
            unsigned int five = 0;
            for (int j = 0; j < M * N; j++)
            {
                if (den_value_c[j] != 124) {
                    printf("den_value_c[%d]=%f d_B_value[%d]=%f thread_id=%u\n", j, den_value_c[j], j, d_B_value[j],
                           (unsigned int) pthread_self());
                    five++;
                    break;
                }
            }
            if (five) {
                usleep(1000);
                abort();
                return NULL;
            }
        }
        #endif
        //printf("kernel five  thread_id=%u\n",(unsigned int)pthread_self());
        ///hipLaunchKernelGGL(set_B_to_256, gridsize, blocksize, 0, temp , d_B_value, M * N);
        set_B_to_256<<<gridsize, blocksize, 0, temp>>> (d_B_value, M * N);

        UPTKStreamSynchronize(temp);
        //usleep(500 * 1);
        #if 0 
        if (256 != d_B_value[M*N/2]){
            printf("------ error after stream:%p five  kernel d_B_value:%f, expect 256,index:%d,line:%d\n",temp, d_B_value[M*N/2],i,__LINE__);
            #if 0
            for(int j=0;j<5; j++){
                printf("\t------ error after stream:%p five  kernel d_B_value[%d]:%f, expect 256,index:%d\n",temp,(j*800),d_B_value[(j*800)],i);
            }
            #endif
            abort();
        }
        #endif
        HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_B_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
        //printf("end kernel five  thread_id=%u\n",(unsigned int)pthread_self());
        flag_thread = check_value(den_value_c,d_B_value, M * N, 256, "B matrix");
        if (!flag_thread) {
            printf("five check error index:%d,stream:%p,den_value_c=%p,d_B_value=%p ,thread_id=%u\n",
                                i,temp,den_value_c,d_B_value, (unsigned int) pthread_self());
            UPTKStreamQuery(temp);
            //HIP_CHECK(UPTKMemcpyAsync(den_value_c, d_B_value, sizeof(float) * M * N, UPTKMemcpyDeviceToHost, temp));
            unsigned five2 = 0;
            for (int j = 0; j < M * N; j++)
            {
                if (den_value_c[j] != 256) {
                    printf("den_value_c[%d]=%f d_B_value[%d]=%f thread_id=%u\n", j, den_value_c[j], j, d_B_value[j],
                           (unsigned int) pthread_self());
                    //break;
                    five2++;
                    break;
                }
            }
            if (five2) {
                usleep(1000);
               abort();
               return NULL;
            }
        }
        //printf("end index:%d thread_id=%u, stream:%p\n",i,(unsigned int)pthread_self(),StreamData);
        // for (int i=0;i<M*N;i++)
        // std::cout<<den_value_c[i]<<" ";
        #if 1 
        UPTKFree(d_A_value);
        UPTKFree(d_B_value);
        UPTKFree(d_C_value);
        UPTKFreeHost(SMM_value);
        UPTKFreeHost(den_value_b);
        UPTKFreeHost(den_value_c);
        //free(SMM_value);
        //free(den_value_b);
        //free(den_value_c);
        #endif
    }
    #if 0 
    UPTKFree(d_A_value);
    UPTKFree(d_B_value);
    UPTKFree(d_C_value);
    UPTKFreeHost(SMM_value);
    UPTKFreeHost(den_value_b);
    UPTKFreeHost(den_value_c);
    #endif
    return NULL;
}

#define THREAD_NUM  5 

TEST(cudaKernel,cudaKerneltest2){
    //创建一个流队列，有两个流
    //UPTKStream_t *queue = (UPTKstream *)malloc(sizeof(UPTKstream)*2);
    //UPTKStreamCreate(&queue[0]);
    //UPTKStreamCreate(&queue[1]);
    int deviceCnt = 0;

    if (UPTKGetDeviceCount(&deviceCnt)) {
        printf("get device count error\n");
        //return 0;
    }
    printf("---- device count:%d\n", deviceCnt);
    pthread_t tid[THREAD_NUM];
    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_create(&tid[i], NULL, csr_mm, (void *) &i);
        //pthread_create(&tid[i], NULL, csr_mm, (void *)&queue[i]);
    }

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(tid[i], NULL);
    }

    //return 0;
}
}
