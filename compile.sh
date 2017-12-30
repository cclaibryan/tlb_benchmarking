arch=$1
nvcc -arch=sm_$arch -o tlb_GPU_$arch tlb_GPU.cu 
