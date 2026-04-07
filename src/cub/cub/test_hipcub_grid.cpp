/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "common_test_header.hpp"

#include "cub/block/block_reduce.cuh"
#include "cub/thread/thread_operators.cuh"

#include "cub/grid/grid_barrier.cuh"
#include "cub/grid/grid_even_share.cuh"
#include "cub/grid/grid_queue.cuh"

__global__ void KernelGridBarrier(
    cub::GridBarrier global_barrier,
    int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        global_barrier.Sync();
    }
}
 
TEST(HipcubGridTests, GridBarrier)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    constexpr int32_t block_size = 256;
    // NOTE increasing iterations will cause huge latency for tests
    constexpr int32_t iterations = 3;
    int32_t grid_size = -1;

    int32_t sm_count;
    int32_t max_block_threads;
    int32_t max_sm_occupancy;

    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_block_threads, cudaDevAttrMaxThreadsPerBlock, device_id));

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_sm_occupancy,
        KernelGridBarrier,
        HIPCUB_HOST_WARP_THREADS,
        0));

    int32_t occupancy = std::min((max_block_threads / block_size), max_sm_occupancy);

    if (grid_size == -1)
    {
        grid_size = occupancy * sm_count;
    }
    else
    {
        occupancy = grid_size / sm_count;
    }

    cub::GridBarrierLifetime global_barrier;
    CUDA_CHECK(global_barrier.Setup(grid_size));

    KernelGridBarrier<<<grid_size, block_size>>>(global_barrier, iterations);
}

template<
    int32_t BlockSize,
    class T,
    typename OffsetT
>
__global__ void KernelGridEvenShare(
    T* device_output,
    T* device_output_reductions,
    cub::GridEvenShare<OffsetT>  even_share)
{
    using breduce_t = cub::BlockReduce<T, BlockSize>;
    __shared__ typename breduce_t::TempStorage temp_storage;

    even_share.template BlockInit<BlockSize, cub::GRID_MAPPING_RAKE>();

    const int32_t index = even_share.block_offset + threadIdx.x;
    if(index > even_share.block_end)
    {
        return;
    }

    T value = device_output[index];

    value = breduce_t(temp_storage).Reduce(value, cub::Sum());
    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = value;
    }
}

TEST(HipcubGridTests, GridEvenShare)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using OffsetT = int32_t;
    using T = uint32_t;
    constexpr size_t block_size = 256;
    constexpr size_t size = block_size * 113;
    constexpr size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        CUDA_CHECK(cudaMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        cub::GridEvenShare<OffsetT> even_share;
        even_share.DispatchInit(size, grid_size, block_size);

        KernelGridEvenShare<block_size, T, OffsetT>
            <<<grid_size, block_size>>>
                (device_output,
                 device_output_reductions,
                 even_share);

        // Reading results back
        CUDA_CHECK(
            cudaMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_reductions));
    }
}

template<typename OffsetT>
__global__ void KernelGridQueueInit(cub::GridQueue<OffsetT> tile_queue)
{
    if ((threadIdx.x == 0) && (blockIdx.x == 0))
    {
        tile_queue.ResetDrain();
    }
}

template<
    int32_t BlockSize,
    class T,
    typename OffsetT
>
__global__ void KernelGridQueue(
    T* device_output,
    T* device_output_reductions,
    OffsetT num_tiles,
    cub::GridQueue<OffsetT> tile_queue)
{
    using breduce_t = cub::BlockReduce<T, BlockSize>;
    __shared__ typename breduce_t::TempStorage temp_storage;
    __shared__ int32_t block_tile_index;

    if(threadIdx.x == 0)
    {
        block_tile_index = tile_queue.Drain(1);
    }
    __syncthreads();

    if(block_tile_index > num_tiles || block_tile_index < 0)
    {
        return;
    }

    int32_t index = block_tile_index * BlockSize + threadIdx.x;
    T value = device_output[index];
    value = breduce_t(temp_storage).Reduce(value, cub::Sum());

    if(threadIdx.x == 0)
    {
        device_output_reductions[block_tile_index] = value;
    }
}

TEST(HipcubGridTests, GridQueue)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using OffsetT = int32_t;
    using T = uint32_t;
    constexpr size_t block_size = 256;
    constexpr size_t size = block_size * 113;
    constexpr size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size);

        // Calculate expected results on host
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            T value = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                value += output[idx];
            }
            expected_reductions[i] = value;
        }

        // Preparing device
        T* device_output;
        CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(T)));
        T* device_output_reductions;
        CUDA_CHECK(cudaMalloc(&device_output_reductions, output_reductions.size() * sizeof(T)));

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        OffsetT* queue_allocations;
        CUDA_CHECK(cudaMalloc(&queue_allocations, cub::GridQueue<OffsetT>().AllocationSize()));
        cub::GridQueue<OffsetT> tile_queue(queue_allocations);

        KernelGridQueueInit<OffsetT><<<1, 1>>>(tile_queue);

        KernelGridQueue<block_size, T, OffsetT>
            <<<grid_size, block_size>>>
                (device_output,
                 device_output_reductions,
                 113,
                 tile_queue);

        CUDA_CHECK(
            cudaMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_reductions));
    }
}
