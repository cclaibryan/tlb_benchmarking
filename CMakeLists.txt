cmake_minimum_required(VERSION 3.8)
project(tlb_benchmark)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        130MB/TLB_in_block.log
        130MB/TLB_in_grid.log
        130MB/TLB_in_warp.log
        2048MB/TLB_in_block.log
        2048MB/TLB_in_grid.log
        2048MB/TLB_in_warp.log
        2MB/TLB_in_block.log
        2MB/TLB_in_grid.log
        2MB/TLB_in_warp