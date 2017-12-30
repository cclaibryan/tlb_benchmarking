#include <iostream>
#include <iomanip>
#include <sys/time.h>
using namespace std;

#define KB 1024/sizeof(unsigned)

#define CHASING		myPtr = h_a[myPtr];
#define CHASING_16  CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING CHASING
#define CHASING_256 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16 CHASING_16
#define CHASING_4096 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256 CHASING_256
#define CHASING_65536 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096 CHASING_4096

double diffTime(struct timeval end, struct timeval start);
void measure_tlb() ;
double para_measure_tlb(int dataSize, int stride);

/*
TLB on Haswell:
	L1:
		4K: 64, 4-way, 2M/4M: 32, 4-way, 1G: 4, 4-way
	L2:
		4K + 2M shared, 1024, 8-way
*/
void measure_tlb() {
	
	int dataSize, stride, beginSize, endSize;

	stride = 4 * KB;	//4K stride
	beginSize = 2 * KB;	//2K
	endSize = 20 * KB;	//6K
	
	int skip = 2 * KB;
	for(dataSize = beginSize; dataSize < endSize; dataSize += skip) {

		if(dataSize > 1 * 1024 * 256)	//larger than 1 MB
    		cout<<sizeof(unsigned)*(float)dataSize/1024/1024<<" MB: ";
    	else
    		cout<<sizeof(unsigned)*(float)dataSize/1024<<" KB: ";

		//warp up run
		para_measure_tlb(dataSize, stride);

    	// cout<<"Stride = "<<stride * sizeof(unsigned)/1024<<" KB, ";
    	double aveTime = para_measure_tlb(dataSize, stride);
    	cout<<setprecision(6)<<aveTime*1000000<<" ns."<<endl;
    	// cout<<"Accessing 65536 times. Average access latency: "<<aveTime<<" ms."<<endl;
	}
}

double para_measure_tlb(int dataSize, int stride) {

	unsigned *h_a = new unsigned[dataSize+2];

	for(unsigned i = 0; i < dataSize; i++) {
		h_a[i] = (i+stride) % dataSize;
	}
	h_a[dataSize] = 0;
	h_a[dataSize+1] = 0;

	int warm_up = dataSize/stride;
	unsigned myPtr = 0;
	struct timeval startTime, endTime;
	double totalTime = 0;
	//warm up iterations
	for(int i = 0; i < warm_up; i++) {
		myPtr = h_a[myPtr];
	}

	int iterations = 10;
	for(int e = 0; e < iterations; e++) {
		gettimeofday(&startTime, NULL);
		CHASING_65536
		gettimeofday(&endTime, NULL);
		double curTime = diffTime(endTime, startTime);
		totalTime += curTime; 
	}
	
	delete[] h_a;
	
	return totalTime / iterations / 65536;
}

//return ms
double diffTime(struct timeval end, struct timeval start) {
	return 1000 * (end.tv_sec - start.tv_sec) + 0.001 * (end.tv_usec - start.tv_usec);
}

int main(int argc, char* argv[]) {
	measure_tlb();
	return 0;
}