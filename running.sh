arch=$1
file=$2
for ((i=1;i<=64;i++));
do
    ./tlb_GPU_$arch 1024 $i 2 128 >> logs/$file/MultiThread/TLB_in_grid_2MB.log
    ./tlb_GPU_$arch 1024 $i 130 2048 >> logs/$file/MultiThread/TLB_in_grid_130MB.log
    ./tlb_GPU_$arch 1024 $i 2048 2048 >> logs/$file/MultiThread/TLB_in_grid_2048MB.log
    ./tlb_GPU_$arch 1024 $i 4096 2048 >> logs/$file/MultiThread/TLB_in_grid_4096MB.log
done

for ((i=32;i<=1024;i+=32));
do
    ./tlb_GPU_$arch $i 1 2 128 >> logs/$file/MultiThread/TLB_in_block_2MB.log
    ./tlb_GPU_$arch $i 1 130 2048 >> logs/$file/MultiThread/TLB_in_block_130MB.log
    ./tlb_GPU_$arch $i 1 2048 2048 >> logs/$file/MultiThread/TLB_in_block_2048MB.log
    ./tlb_GPU_$arch $i 1 4096 2048 >> logs/$file/MultiThread/TLB_in_block_4096MB.log
done




