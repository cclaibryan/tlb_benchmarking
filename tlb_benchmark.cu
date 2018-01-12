#include <iostream>
#include <set>
#include <algorithm>
#include <assert.h>
#include "cuda_runtime.h"
using namespace std;

#define ITERATION_FINEGRAINED   (1)
#define KB                      (1024/sizeof(int))
#define MB                      (KB*1024)
#define MAX_NUM_THREADS         (1024)      // a block has maximal thread size

//kernel function
__global__ void strided_access_onepass(unsigned *arr, int length, int stride, bool record, unsigned *duration, double *help);   //used to benchmark the TLB structure
__global__ void strided_access_finegrained(unsigned *arr, int length, bool record, unsigned *duration, unsigned *index);        //obsolete: use to attain average cycle and pages visited

void TLB_latency(int N, int stride);
void TLB_benchmarking(int beginSize, int endSize, int stride);

void generate_strided(unsigned *arr, int length, int stride);
void generate_strided_onepass(unsigned *arr, int length, int stride);

//global variables
int numThreadsGlobal;
int numBlocksGlobal;
int dataSizeGlobal;   //in MB
int pageSizeGlobal;   //in KB

/*
 * TLB benchmarking: ./tlb_GPU pageSize_KB dataSize_begin_MB dataSize_end_MB
 *
 * blockSize=1 and gridSize=1 for TLB benchmarking;
 */
int main(int argc, char* argv[]){

    if (argc < 4)  {
        cerr<<"Shall provide the blockSize, gridSize used and page size."<<endl;
        cerr<<"Eg.: ./tlb_GPU bSize gSize dataSize_MB pageSize_KB"<<endl;
        exit(0);
    }

    numThreadsGlobal = 1;
    numBlocksGlobal = 1;
    pageSizeGlobal = atoi(argv[1]) * KB;
    int dataSize_begin = atoi(argv[2]) * MB;
    int dataSize_end = atoi(argv[3]) * MB;
    cudaSetDevice(0);

    TLB_benchmarking(dataSize_begin, dataSize_end,pageSizeGlobal);

    cudaDeviceReset();
    return 0;
}

void TLB_benchmarking(int beginSize, int endSize, int stride) {

    for (int ds = beginSize; ds <= endSize; ds += stride) {
        cout << "Struc: Data size: " << (float)ds / MB << "MB\t" << "Stride: " << stride / MB << "MB\t";

        cudaDeviceReset();
        cudaError_t error_id;
        unsigned *h_a, *d_a;
        unsigned *h_timeinfo, *d_timeinfo;
        double *help;

        h_a = (unsigned*)malloc(sizeof(unsigned)*ds);
        error_id = cudaMalloc ((void **) &d_a, sizeof(unsigned)*ds);
        if (error_id != cudaSuccess)    cerr<<"Error 1.0 is "<<cudaGetErrorString(error_id)<<endl;

        /* initialize array elements on CPU with pointers into d_a. */
        generate_strided_onepass(h_a,ds,stride);

        /* copy array elements from CPU to GPU */
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned)*ds, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess)    cerr<<"Error 1.1 is "<<cudaGetErrorString(error_id)<<endl;

        h_timeinfo = (unsigned *) malloc(sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
        error_id = cudaMalloc((void **) &d_timeinfo, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
        if (error_id != cudaSuccess)    cerr << "Error 1.2 is " << cudaGetErrorString(error_id) << endl;

        error_id = cudaMalloc((void **) &help, sizeof(double) * numThreadsGlobal * numBlocksGlobal);
        if (error_id != cudaSuccess)    cerr << "Error 1.3 is " << cudaGetErrorString(error_id) << endl;

        cudaThreadSynchronize();
        /* launch kernel*/
        dim3 Db = dim3(numThreadsGlobal);
        dim3 Dg = dim3(numBlocksGlobal);

        strided_access_onepass<<< Dg, Db >>> (d_a, ds, stride, false, NULL, NULL);        //warp up
        strided_access_onepass<<< Dg, Db >>> (d_a, ds, stride, true, d_timeinfo, help);   //formal

        cudaThreadSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            cerr << "Error kernel is " << cudaGetErrorString(error_id) << endl;
        }

        /* copy results from GPU to CPU */
        cudaThreadSynchronize();

        error_id = cudaMemcpy((void *) h_timeinfo, (void *) d_timeinfo, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess)    cerr << "Error 2.2 is " << cudaGetErrorString(error_id) << endl;

        double total = 0;       //here we use double, otherwise it will overflow
        for (int i = 0; i < numThreadsGlobal * numBlocksGlobal; i++) {
            total += h_timeinfo[i];
        }
        total /= (numThreadsGlobal * numBlocksGlobal);
        cout << "cycle: " << total << endl;

        cudaThreadSynchronize();

        /* free memory on GPU */
        cudaFree(help);
        cudaFree(d_a);
        cudaFree(d_timeinfo);

        /*free memory on CPU */
        free(h_a);
        free(h_timeinfo);

        cudaDeviceReset();
    }
}

//used for TLB benchmarking
__global__ void strided_access_onepass(unsigned *arr, int length, int stride, bool record, unsigned *duration, double *help) {

    unsigned long start, end;
    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned curIdx = 0;

    double anc = 0;
    double total = 0;
    int myIteration = 0;

    //traverse the data array once
    while (curIdx < length) {
        start = clock64();
            curIdx = arr[curIdx];
            anc += curIdx;                  //to ensure the curIdx has been read, this instruction is 16-cycle long on K40m
        end = clock64();
        total += (end-start-16);
        myIteration++;
    }

    if (record)     {
        duration[gid] = (total/myIteration);
        help[gid] = anc;
    }
}

void generate_strided_onepass(unsigned *arr, int length, int stride) {
    for (int i = 0 ; i < length; i++) {
        arr[i] = i+stride;
    }
}

//void measure_global() {
//
//    int stride = pageSizeGlobal*KB; //2MB stride
//    set<int> missPages; //recording the overall missing pages in each case
//
//    //begin and end size in MBs
//    /* To test the TLB structures the beginSize and endSize is different;
//     * To test the latency of multi-thread, beginSize and endSize should set as the data size tested */
//    int beginSize = dataSizeGlobal * MB;
//    int endSize = dataSizeGlobal * MB;
//
//    //1. The L1 TLB has 16 entries. Test with N_min=28 *1024*256, N_max>32*1024*256
//    //2. The L2 TLB has 65 entries. Test with N_min=128*1024*256, N_max=160*1024*256
//    for (int dataSize = beginSize; dataSize <= endSize; dataSize += (128*KB)) {
////        cout<<"Data size: "<<(float)dataSize/MB<<"MB\t"<<"Stride: "<< stride/MB <<"MB"<<endl;
//        cout<<"Data size: "<<(float)dataSize/MB<<"MB\tbsize: "<<numThreadsGlobal<<"\tgsize: "<<numBlocksGlobal<<'\t';
//        parametric_measure_global(dataSize, false, stride, missPages);  //not finegrained
//    }
//}
//void TLB_finegrained(int N, bool finegrained, int stride, set<int> & lastMissPages) {
//    cudaDeviceReset();
//    cudaError_t error_id;
//    int i;
//    unsigned *h_a, *d_a;
//    h_a = (unsigned*)malloc(sizeof(unsigned)*N);
//    error_id = cudaMalloc ((void **) &d_a, sizeof(unsigned)*N);
//
//    if (error_id != cudaSuccess)
//        cerr<<"Error 1.0 is "<<cudaGetErrorString(error_id)<<endl;
//
//    /* initialize array elements on CPU with pointers into d_a. */
//    generate_strided(h_a,N,stride);
//    //generate_strided_onepass(h_a,N,(mul)*stride);
//
//    /* copy array elements from CPU to GPU */
//    error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned)*N, cudaMemcpyHostToDevice);
//    if (error_id != cudaSuccess) {
//        cerr<<"Error 1.1 is "<<cudaGetErrorString(error_id)<<endl;
//    }
//
//    unsigned  *h_index, *h_timeinfo, *d_timeinfo, *d_index;
//    double *help;
//
//    if (finegrained) {
//        h_index = (unsigned *) malloc(sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal * ITERATION);
//        h_timeinfo = (unsigned *) malloc(sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal * ITERATION);
//
//        //recording time and visited locations
//        error_id = cudaMalloc((void **) &d_timeinfo, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal * ITERATION);
//        if (error_id != cudaSuccess) {
//            cerr << "Error 1.2 is " << cudaGetErrorString(error_id) << endl;
//        }
//
//        error_id = cudaMalloc((void **) &d_index, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal *ITERATION);
//        if (error_id != cudaSuccess) {
//            cerr << "Error 1.3 is " << cudaGetErrorString(error_id) << endl;
//        }
//    }
//    else {
//        h_timeinfo = (unsigned *) malloc(sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
//        error_id = cudaMalloc((void **) &d_timeinfo, sizeof(unsigned) * numThreadsGlobal * numBlocksGlobal);
//        if (error_id != cudaSuccess) {
//            cerr << "Error 1.4 is " << cudaGetErrorString(error_id) << endl;
//        }
//        error_id = cudaMalloc((void **) &help, sizeof(double) * numThreadsGlobal * numBlocksGlobal);
//        if (error_id != cudaSuccess) {
//            cerr << "Error 1.5 is " << cudaGetErrorString(error_id) << endl;
//        }
//    }
//
//    cudaThreadSynchronize ();
//    /* launch kernel*/
//    dim3 Db = dim3(numThreadsGlobal);
//    dim3 Dg = dim3(numBlocksGlobal);
//    if (finegrained) {
//        strided_access_finegrained<<<Dg, Db>>>(d_a, N, false, NULL, NULL);
//        strided_access_finegrained<<<Dg, Db>>>(d_a, N, false, d_timeinfo, d_index);
//    }
//    else {
//        strided_access<<<Dg, Db>>>(d_a, N, stride, false, NULL, NULL);        //warp up
//        strided_access<<<Dg, Db>>>(d_a, N, stride, true, d_timeinfo, help);   //formal
//    }
//
//    cudaThreadSynchronize();
//
//    error_id = cudaGetLastError();
//    if (error_id != cudaSuccess) {
//        cerr<<"Error kernel is "<<cudaGetErrorString(error_id)<<endl;
//    }
//
//    /* copy results from GPU to CPU */
//    cudaThreadSynchronize ();
//
//    if (finegrained) {
//        error_id = cudaMemcpy((void *)h_timeinfo, (void *)d_timeinfo, sizeof(unsigned)*ITERATION*numThreadsGlobal * numBlocksGlobal, cudaMemcpyDeviceToHost);
//        if (error_id != cudaSuccess) {
//            cerr<<"Error 2.0 is "<<cudaGetErrorString(error_id)<<endl;
//        }
//        error_id = cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned)*ITERATION*numThreadsGlobal * numBlocksGlobal, cudaMemcpyDeviceToHost);
//        if (error_id != cudaSuccess) {
//            cerr<<"Error 2.1 is "<<cudaGetErrorString(error_id)<<endl;
//        }
//
//        //statistics
//        int count_less_300 = 0, count_300_400 = 0, count_400_500 = 0, count_500_600 = 0, count_larger_600 = 0;
//        double total = 0;
//
//        int loop = 0;           //how many times the array is looped
//
//        set<int> curMissPages;
//        for(i=0 ;i<ITERATION;i++) {
//            int curPage = h_index[i]/stride;
//            if ( (h_timeinfo[i] > 400) && (h_timeinfo[i] < 510)) {
//                curMissPages.insert(curPage);
//            }
//            cout<<curPage<<'\t'<<h_index[i]<<'\t'<<h_timeinfo[i]<<endl;
//
//            if (h_index[i]<stride)  loop ++;
//            if (h_timeinfo[i] < 300)            count_less_300++;
//            else if (h_timeinfo[i] < 400) count_300_400 ++;
//            else if (h_timeinfo[i] < 500) count_400_500 ++;
//            else if (h_timeinfo[i] < 600) count_500_600++;
//            else                                                    count_larger_600++;
//            total += h_timeinfo[i];
//        }
//        set<int> diffSet;
//        set_difference(curMissPages.begin(), curMissPages.end(),lastMissPages.begin(), lastMissPages.end(), inserter(diffSet,diffSet.end()));
//
//        //to check that pages missed in last dataset will be hit in this dataset
//        set<int> checkSet;
//        set_difference(lastMissPages.begin(), lastMissPages.end(), curMissPages.begin(), curMissPages.end(), inserter(checkSet,checkSet.end()));
//        assert(checkSet.size() == 0);
//
//        int totalPages = N /512 / 1024;
//        cout<<"Pages: "<<totalPages<<", misses: "<<count_400_500<<", loops: "<<loop<<", new miss pages: ";
//        for (set<int>::iterator it = diffSet.begin(); it != diffSet.end(); ++it) {
//            cout<<*it<<' ';
//        }
//        cout<<endl;
//
//        // lastMissPages = curMissPages;
//
//        total = total / ITERATION;
//        cout<<"Average: "<<total<<endl;
//        // cout<<"Statistics:"<<endl;
//        // cout<<"Data size: "<<N / 1024 / 256<<" MB."<<endl;
//
//        // cout<<"less than 300: "<<count_less_300<<endl;
//        // cout<<"300 - 400: "<<count_300_400<<endl;
//        // cout<<"400 - 500: "<<count_400_500<<endl;
//        // cout<<"500 - 600: "<<count_500_600<<endl;
//        // cout<<"larger than 600: "<<count_larger_600<<endl;
//        // cout<<"Average cycles: "<<total<<" in "<<ITERATION<<" iterations."<<endl;
//    }
//    else {
//        error_id = cudaMemcpy((void *)h_timeinfo, (void *)d_timeinfo, sizeof(unsigned)*numThreadsGlobal * numBlocksGlobal, cudaMemcpyDeviceToHost);
//        if (error_id != cudaSuccess) {
//            cerr<<"Error 2.2 is "<<cudaGetErrorString(error_id)<<endl;
//        }
//
//        double total = 0;       //here we use double, otherwise it will overflow
//        for(int i = 0; i < numThreadsGlobal*numBlocksGlobal; i++) {
//            total += h_timeinfo[i];
//        }
//        total /= (numThreadsGlobal*numBlocksGlobal);
//        cout<<"cycle: "<<total<<endl;
//    }
//    cudaThreadSynchronize();
//
//    /* free memory on GPU */
//    if (finegrained) {
//        cudaFree(d_index);
//        free(h_index);
//    }
//    else {
//        cudaFree(help);
//    }
//
//    cudaFree(d_a);
//    cudaFree(d_timeinfo);
//
//    /*free memory on CPU */
//    free(h_a);
//    free(h_timeinfo);
//
//    cudaDeviceReset();
//}
//obsolete: to record the page number and study the cache replacement policy
//__global__ void strided_access_finegrained(unsigned *arr, int length, bool record, unsigned *duration, unsigned *index) {
//
//    unsigned timestamp;
//    unsigned gid = blockDim.x * blockIdx.x + threadIdx.x;
//    unsigned gsize = blockDim.x * gridDim.x;
//    unsigned curIdx = (blockDim.x * threadIdx.x + blockIdx.x) % length;
//
//    __shared__ unsigned int s_tvalue[ITERATION_FINEGRAINED*MAX_NUM_THREADS];
//    __shared__ unsigned int s_index[ITERATION_FINEGRAINED*MAX_NUM_THREADS];
//
//    unsigned it = gid;
//    while (it < ITERATION_FINEGRAINED * MAX_NUM_THREADS) {
//        s_index[it] = 0;
//        s_tvalue[it] = 0;
//        it += gsize;
//    }
//    __syncthreads();
//
//    it = gid;
//    for (int k = 0; k < ITERATION_FINEGRAINED; k++) {
//        timestamp = clock();
//        curIdx = arr[curIdx];
//        s_index[it]= curIdx;
//        timestamp = clock() - timestamp;
//        s_tvalue[it] = timestamp;
//        it += ITERATION_FINEGRAINED;
//    }
//
//    if (record) {
//        it = threadIdx.x;
//        while (it < blockDim.x * ITERATION_FINEGRAINED) {
//            duration[it + blockIdx.x*blockDim.x*ITERATION_FINEGRAINED] = s_tvalue[it];
//            index[it + blockIdx.x*blockDim.x*ITERATION_FINEGRAINED] = s_index[it];
//            it += blockDim.x;
//        }
//    }
//}