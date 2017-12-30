for ((i=1;i<=64;i++));
do
    ./tlb_GPU_128KB_stride 1024 $i 2 >> 2MB/TLB_in_grid.log
    ./tlb_GPU_2MB_stride 1024 $i 130 >> 130MB/TLB_in_grid.log
    ./tlb_GPU_2MB_stride 1024 $i 2048 >> 2048MB/TLB_in_grid.log
    ./tlb_GPU_2MB_stride 1024 $i 4096 >> 4096MB/TLB_in_grid.log
done

for ((i=32;i<=1024;i+=32));
do
    ./tlb_GPU_128KB_stride $i 1 2 >> 2MB/TLB_in_block.log
    ./tlb_GPU_2MB_stride $i 1 130 >> 130MB/TLB_in_block.log
    ./tlb_GPU_2MB_stride $i 1 2048 >> 2048MB/TLB_in_block.log
    ./tlb_GPU_2MB_stride $i 1 4096 >> 4096MB/TLB_in_block.log
done

for ((i=1;i<=32;i++));
do
    ./tlb_GPU_128KB_stride $i 1 2 >> 2MB/TLB_in_warp.log
    ./tlb_GPU_2MB_stride $i 1 130 >> 130MB/TLB_in_warp.log
    ./tlb_GPU_2MB_stride $i 1 2048 >> 2048MB/TLB_in_warp.log
    ./tlb_GPU_2MB_stride $i 1 4096 >> 4096MB/TLB_in_warp.log
done




