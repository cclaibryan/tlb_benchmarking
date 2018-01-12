arch=$1
nvcc -arch=sm_$arch -o tlb_latency_$arch tlb_latency.cu
nvcc -arch=sm_$arch -o tlb_benchmark_$arch tlb_benchmark.cu

