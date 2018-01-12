arch=$1
file=$2
for ((i=1;i<=100;i++));
do
    ./tlb_latency_$arch 1024 $i 2048 32  >> logs/$file/MultiThread/TLB_in_grid_32MB.log
    ./tlb_latency_$arch 1024 $i 32768 2048  >> logs/$file/MultiThread/TLB_in_grid_2048MB.log
    ./tlb_latency_$arch 1024 $i 32768 4096  >> logs/$file/MultiThread/TLB_in_grid_4096MB.log
done

for ((i=32;i<=1024;i+=32));
do
    ./tlb_latency_$arch $i 1 2048 32  >> logs/$file/MultiThread/TLB_in_block_32MB.log
    ./tlb_latency_$arch $i 1 32768 2048  >> logs/$file/MultiThread/TLB_in_block_2048MB.log
    ./tlb_latency_$arch $i 1 32768 4096  >> logs/$file/MultiThread/TLB_in_blok_4096MB.log
done




