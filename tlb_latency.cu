#include <iostream>
#include <set>
#include <algorithm>
#include <assert.h>
#include "cuda_runtime.h"
using namespace std;

#define ITERATIONS              (10000)           //times of memory visit for each thread
#define KB                      (1024/sizeof(int))
#define MB                      (KB*1024)
#define MAX_NUM_THREADS         (1024)      // a block has maximal thread size
#define EXPER_TIME              (10)        //experiments are repeated 10 times

//kernel function
__global__ void strided_access(unsigned *arr, int length, int stride, bool record, unsigned *duration, unsigned *help);           //used to attain the average cycle of the multi-threaded kernel

void TLB_latency(int N, int stride);
void generate_strided(unsigned *arr, int length, int stride);

//global variables
int numThreadsGlobal;
int numBlocksGlobal;
int dataSizeGlobal;   //in MB
int pageSizeGlobal;   //in KB

/*
 * TLB latency: ./tlb_GPU blockSize gridSize pageSize_KB dataSize_MB
 *
 * others for TLB latency.
 */
int main(int argc, char* argv[]){

    if (argc < 4)  {
        cerr<<"Shall provide the blockSize, gridSize used and page size."<<endl;
        cerr<<"Eg.: ./tlb_GPU bSize gSize pageSize_KB dataSize_MB"<<endl;
        exit(0);
    }

    numThreadsGlobal = atoi(argv[1]);
    numBlocksGlobal = atoi(argv[2]);
    pageSizeGlobal = atoi(argv[3]) * KB;
    dataSizeGlobal = atoi(argv[4]) * MB;

    cudaSetDevice(0);
    cout<<"Latency: Data size: "<<(float)dataSizeGlobal/MB<<"MB\tbsize: "<<numThreadsGlobal<<"\tgsize: "<<numBlocksGlobal<<'\t';
    TLB_latency(dataSizeGlobal, pageSizeGlobal);
    cudaDeviceReset();
    return 0;
}

//multi-threaded kernels
__global__ void strided_access(unsigned *arr, int length, int stride, bool record, unsigned *duration, unsigned *help) {

    unsigned long timestamp;
    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned curIdx = (gid* (stride+100)) % length;         //adding an offset in case that only a few elements are accessed

    unsigned anc = 0;
    double total = 0;

    //repeated visit, run fixed iterations
    timestamp = clock64();
    for (int i = 0; i < ITERATIONS; i++) {
        curIdx = arr[curIdx];
        anc += curIdx;                  //to ensure the curIdx has been read, this instruction is 16-cycle long on K40m
    }
    timestamp = clock64() - timestamp;
    total += timestamp;

    if (record)     {
        duration[gid] = total/ITERATIONS-16;    //deduce the register add instruction overhead
        help[gid] = anc;
    }
}

/*
 * N: number of data elements
 * stride: stride for strided-access, set as the page size
 */
void TLB_latency(int N, int stride) {

    cudaDeviceReset();
    cudaError_t error_id;
    unsigned *h_a, *d_a;
    unsigned *h_timeinfo, *d_timeinfo;
    unsigned *help;

    h_a = (unsigned*)malloc(sizeof(unsigned)*N);
    error_id = cudaMalloc ((void **) &d_a, sizeof(unsigned)*N);
    if (error_id != cudaSuccess)    cerr<<"Error 1.0 is "<<cudaGetErrorString(error_id)<<endl;

    /* initialize array elements on CPU with pointers into d_a. */
    generate_strided(h_a,N,stride);

    /* copy array elements from CPU to GPU */
    error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned)*N, cudaMemcpyHostToDevice);
    if (error_id != cudaSuccess)    cerr<<"Error 1.1 is "<<cudaGetErrorString(error_id)<<endl;

    h_timeinfo = (unsigned *) malloc(sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
    error_id = cudaMalloc((void **) &d_timeinfo, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
    if (error_id != cudaSuccess)    cerr << "Error 1.2 is " << cudaGetErrorString(error_id) << endl;

    error_id = cudaMalloc((void **) &help, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
    if (error_id != cudaSuccess)    cerr << "Error 1.3 is " << cudaGetErrorString(error_id) << endl;

    cudaThreadSynchronize();

    dim3 Db = dim3(numThreadsGlobal);
    dim3 Dg = dim3(numBlocksGlobal);

    double total = 0;
    /* launch kernel*/
    for (int e = 0; e < EXPER_TIME; e++) {
        //kernel execution
        strided_access<<<Dg, Db>>>(d_a, N, stride, false, NULL, NULL);        //warp up
        strided_access<<<Dg, Db>>>(d_a, N, stride, true, d_timeinfo, help);   //recording
        cudaThreadSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess)    cerr<<"Error kernel is "<<cudaGetErrorString(error_id)<<endl;

        /* copy results from GPU to CPU */
        cudaThreadSynchronize ();
        error_id = cudaMemcpy((void *)h_timeinfo, (void *)d_timeinfo, sizeof(unsigned)*numThreadsGlobal * numBlocksGlobal, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess)    cerr<<"Error 2.2 is "<<cudaGetErrorString(error_id)<<endl;

        double temp = 0;       //here we use double, otherwise it will overflow
        for(int i = 0; i < numThreadsGlobal*numBlocksGlobal; i++) {
            temp += h_timeinfo[i];
        }
        temp /= (numThreadsGlobal*numBlocksGlobal);
        total += temp;

        cudaThreadSynchronize();
    }
    total /= EXPER_TIME;
    cout<<"cycle: "<<total<<endl;

    /* free memory on GPU */
    cudaFree(help);
    cudaFree(d_a);
    cudaFree(d_timeinfo);

    /*free memory on CPU */
    free(h_a);
    free(h_timeinfo);

    cudaDeviceReset();
}

void generate_strided(unsigned *arr, int length, int stride) {
    for (int i = 0; i < length; i++) {
        //arr[i] = (i + stride) % length;
        arr[i] = (i + 2048*1024/256) % length;
    }
}