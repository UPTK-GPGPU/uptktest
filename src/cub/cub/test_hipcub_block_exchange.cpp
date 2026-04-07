// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "common_test_header.hpp"

// hipcub API
#include "cub/block/block_store.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"
#include "cub/block/block_exchange.cuh"

template<
    class T,
    class U,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
struct params
{
    using type = T;
    using output_type = U;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

template<class Params>
class HipcubBlockExchangeTests : public ::testing::Test {
public:
    using params = Params;
};

template<class T>
struct dummy
{
    T x;
    T y;

#ifdef HIPCUB_ROCPRIM_API
    HIPCUB_HOST_DEVICE
#endif
    dummy() = default;

    template<class U>
    HIPCUB_HOST_DEVICE
    dummy(U a) : x(a + 1), y(a * 2) { }

    HIPCUB_HOST_DEVICE
    bool operator==(const dummy& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }
};

typedef ::testing::Types<
    // Power of 2 BlockSize and ItemsPerThread = 1 (no rearrangement)
    params<int, int, 128, 4>,
    params<int, long long, 64, 1>,
    params<unsigned long long, unsigned long long, 128, 1>,
    params<short, dummy<int>, 256, 1>,
    params<long long, long long, 512, 1>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<int, int, 64, 2>,
    params<long long, long long, 256, 4>,
    params<int, int, 512, 5>,
    params<short, dummy<float>, 128, 7>,
    params<int, int, 128, 3>,
    params<unsigned long long, unsigned long long, 64, 3>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<int, double, 33U, 5>,
    params<char, dummy<double>, 464U, 2>,
    params<unsigned short, unsigned int, 100U, 3>,
    params<short, int, 234U, 9>
> Params;

TYPED_TEST_SUITE(HipcubBlockExchangeTests, Params);

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(512)
void blocked_to_striped_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockExchange<Type, block_size, ItemsPerThread> exchange;
    exchange.BlockedToStriped(input, output);

    cub::StoreDirectBlocked(lid, device_output + block_offset, output);
}

TYPED_TEST(HipcubBlockExchangeTests, BlockedToStriped)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset + ii * block_size + ti;
                input[i1] = values[i1];
                expected[i0] = values[i1];
            }
        }
    }

    // Preparing device
    type* device_input;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    CUDA_CHECK(
        cudaMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            cudaMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    blocked_to_striped_kernel<type, output_type, items_per_block, items_per_thread>
        <<< dim3(grid_size), dim3(block_size), 0, 0>>> (
        device_input, device_output
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reading results
    CUDA_CHECK(
        cudaMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            cudaMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(512)
void striped_to_blocked_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockExchange<Type, block_size, ItemsPerThread> exchange;
    exchange.StripedToBlocked(input, output);

    cub::StoreDirectBlocked(lid, device_output + block_offset, output);
}

TYPED_TEST(HipcubBlockExchangeTests, StripedToBlocked)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset + ii * block_size + ti;
                input[i0] = values[i1];
                expected[i1] = values[i1];
            }
        }
    }

    // Preparing device
    type* device_input;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    CUDA_CHECK(
        cudaMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            cudaMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    striped_to_blocked_kernel<type, output_type, items_per_block, items_per_thread> 
        <<<dim3(grid_size), dim3(block_size), 0, 0>>> (
        device_input, device_output
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reading results
    CUDA_CHECK(
        cudaMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            cudaMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(512)
void blocked_to_warp_striped_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockExchange<Type, block_size, ItemsPerThread> exchange;
    exchange.BlockedToWarpStriped(input, output);

    cub::StoreDirectBlocked(lid, device_output + block_offset, output);
}

TYPED_TEST(HipcubBlockExchangeTests, BlockedToWarpStriped)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    // Given block size not supported
    bool is_block_size_unsupported = block_size > test_utils::get_max_block_size();
#ifdef HIPCUB_CUB_API
    // CUB does not support exchanges to/from warp-striped arrangements
    // for incomplete blocks (not divisible by warp size)
    // Workaround for nvcc warning: "dynamic initialization in unreachable code"
    // (not a simple if with compile-time expression)
    is_block_size_unsupported |= block_size % current_device_warp_size != 0;
#endif
    if(is_block_size_unsupported)
    {
        printf("Unsupported test block size: %zu.     Skipping test\n", block_size);
        GTEST_SKIP();
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    constexpr size_t warp_size_32 = test_utils::get_min_warp_size(block_size, size_t(HIPCUB_WARP_SIZE_32));
    constexpr size_t warp_size_64 = test_utils::get_min_warp_size(block_size, size_t(HIPCUB_WARP_SIZE_64));
    constexpr size_t warps_no_32 = (block_size + warp_size_32 - 1) / warp_size_32;
    constexpr size_t warps_no_64 = (block_size + warp_size_64 - 1) / warp_size_64;
    constexpr size_t items_per_warp_32 = warp_size_32 * items_per_thread;
    constexpr size_t items_per_warp_64 = warp_size_64 * items_per_thread;

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);

    const size_t warps_no = current_device_warp_size == HIPCUB_WARP_SIZE_32 ? warps_no_32 : warps_no_64;
    const size_t warp_size = current_device_warp_size == HIPCUB_WARP_SIZE_32 ? warp_size_32 : warp_size_64;
    const size_t items_per_warp = current_device_warp_size == HIPCUB_WARP_SIZE_32 ? items_per_warp_32 : items_per_warp_64;

    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t wi = 0; wi < warps_no; wi++)
        {
            const size_t current_warp_size = wi == warps_no - 1
                ? (block_size % warp_size != 0 ? block_size % warp_size : warp_size)
                : warp_size;
            for(size_t li = 0; li < current_warp_size; li++)
            {
                for(size_t ii = 0; ii < items_per_thread; ii++)
                {
                    const size_t offset = bi * items_per_block + wi * items_per_warp;
                    const size_t i0 = offset + li * items_per_thread + ii;
                    const size_t i1 = offset + ii * current_warp_size + li;
                    input[i1] = values[i1];
                    expected[i0] = values[i1];
                }
            }
        }
    }

    // Preparing device
    type* device_input;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    CUDA_CHECK(
        cudaMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            cudaMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    blocked_to_warp_striped_kernel<
                type, output_type, items_per_block, items_per_thread>
        <<<dim3(grid_size), dim3(block_size), 0, 0>>> (
        device_input, device_output
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reading results
    CUDA_CHECK(
        cudaMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            cudaMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(512)
void warp_striped_to_blocked_kernel(Type* device_input, OutputType* device_output)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockExchange<Type, block_size, ItemsPerThread> exchange;
    exchange.WarpStripedToBlocked(input, output);

    cub::StoreDirectBlocked(lid, device_output + block_offset, output);
}

TYPED_TEST(HipcubBlockExchangeTests, WarpStripedToBlocked)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const unsigned int current_device_warp_size = HIPCUB_HOST_WARP_THREADS;
    // Given block size not supported
    bool is_block_size_unsupported = block_size > test_utils::get_max_block_size();
#ifdef HIPCUB_CUB_API
    // CUB does not support exchanges to/from warp-striped arrangements
    // for incomplete blocks (not divisible by warp size)
    // Workaround for nvcc warning: "dynamic initialization in unreachable code"
    // (not a simple if with compile-time expression)
    is_block_size_unsupported |= block_size % current_device_warp_size != 0;
#endif
    if(is_block_size_unsupported)
    {
        printf("Unsupported test block size: %zu.     Skipping test\n", block_size);
        GTEST_SKIP();
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));

    constexpr size_t warp_size_32 = test_utils::get_min_warp_size(block_size, size_t(HIPCUB_WARP_SIZE_32));
    constexpr size_t warp_size_64 = test_utils::get_min_warp_size(block_size, size_t(HIPCUB_WARP_SIZE_64));
    constexpr size_t warps_no_32 = (block_size + warp_size_32 - 1) / warp_size_32;
    constexpr size_t warps_no_64 = (block_size + warp_size_64 - 1) / warp_size_64;
    constexpr size_t items_per_warp_32 = warp_size_32 * items_per_thread;
    constexpr size_t items_per_warp_64 = warp_size_64 * items_per_thread;

    // Calculate input and expected results on host
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);

    const size_t warps_no = current_device_warp_size == HIPCUB_WARP_SIZE_32 ? warps_no_32 : warps_no_64;
    const size_t warp_size = current_device_warp_size == HIPCUB_WARP_SIZE_32 ? warp_size_32 : warp_size_64;
    const size_t items_per_warp = current_device_warp_size == HIPCUB_WARP_SIZE_32 ? items_per_warp_32 : items_per_warp_64;

    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t wi = 0; wi < warps_no; wi++)
        {
            const size_t current_warp_size = wi == warps_no - 1
                ? (block_size % warp_size != 0 ? block_size % warp_size : warp_size)
                : warp_size;
            for(size_t li = 0; li < current_warp_size; li++)
            {
                for(size_t ii = 0; ii < items_per_thread; ii++)
                {
                    const size_t offset = bi * items_per_block + wi * items_per_warp;
                    const size_t i0 = offset + li * items_per_thread + ii;
                    const size_t i1 = offset + ii * current_warp_size + li;
                    input[i0] = values[i1];
                    expected[i1] = values[i1];
                }
            }
        }
    }

    // Preparing device
    type* device_input;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

    CUDA_CHECK(
        cudaMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            cudaMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    warp_striped_to_blocked_kernel<type, output_type, items_per_block, items_per_thread>
        <<< dim3(grid_size), dim3(block_size), 0, 0>>> (
        device_input, device_output
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reading results
    CUDA_CHECK(
        cudaMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            cudaMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(512)
void scatter_to_blocked_kernel(Type* device_input, OutputType* device_output, unsigned int* device_ranks)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);
    cub::LoadDirectBlocked(lid, device_ranks + block_offset, ranks);

    cub::BlockExchange<Type, block_size, ItemsPerThread> exchange;
    exchange.ScatterToBlocked(input, output, ranks);

    cub::StoreDirectBlocked(lid, device_output + block_offset, output);
}

TYPED_TEST(HipcubBlockExchangeTests, ScatterToBlocked)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));
    std::vector<unsigned int> ranks(size);

    // Calculate input and expected results on host
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, std::mt19937{std::random_device{}()});
    }
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset + ranks[i0];
                input[i0] = values[i0];
                expected[i1] = values[i0];
            }
        }
    }

    // Preparing device
    type* device_input;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    unsigned int* device_ranks;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_ranks, ranks.size() * sizeof(typename decltype(ranks)::value_type)));

    CUDA_CHECK(
        cudaMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            device_ranks, ranks.data(),
            ranks.size() * sizeof(unsigned int),
            cudaMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    scatter_to_blocked_kernel<type, output_type, items_per_block, items_per_thread>
        <<<dim3(grid_size), dim3(block_size), 0, 0>>> (
        device_input, device_output, device_ranks
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reading results
    CUDA_CHECK(
        cudaMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            cudaMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_ranks));
}

template<
    class Type,
    class OutputType,
    unsigned int ItemsPerBlock,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(512)
void scatter_to_striped_kernel(Type* device_input, OutputType* device_output, unsigned int* device_ranks)
{
    constexpr unsigned int block_size = (ItemsPerBlock / ItemsPerThread);
    const unsigned int lid = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * ItemsPerBlock;

    Type input[ItemsPerThread];
    OutputType output[ItemsPerThread];
    unsigned int ranks[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);
    cub::LoadDirectBlocked(lid, device_ranks + block_offset, ranks);

    cub::BlockExchange<Type, block_size, ItemsPerThread> exchange;
    exchange.ScatterToStriped(input, output, ranks);

    cub::StoreDirectBlocked(lid, device_output + block_offset, output);
}

TYPED_TEST(HipcubBlockExchangeTests, ScatterToStriped)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    using output_type = typename TestFixture::params::output_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = items_per_block * 113;
    // Generate data
    std::vector<type> input(size);
    std::vector<output_type> expected(size);
    std::vector<output_type> output(size, output_type(0));
    std::vector<unsigned int> ranks(size);

    // Calculate input and expected results on host
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        auto block_ranks = ranks.begin() + bi * items_per_block;
        std::iota(block_ranks, block_ranks + items_per_block, 0);
        std::shuffle(block_ranks, block_ranks + items_per_block, std::mt19937{std::random_device{}()});
    }
    std::vector<type> values(size);
    std::iota(values.begin(), values.end(), 0);
    for(size_t bi = 0; bi < size / items_per_block; bi++)
    {
        for(size_t ti = 0; ti < block_size; ti++)
        {
            for(size_t ii = 0; ii < items_per_thread; ii++)
            {
                const size_t offset = bi * items_per_block;
                const size_t i0 = offset + ti * items_per_thread + ii;
                const size_t i1 = offset
                    + ranks[i0] % block_size * items_per_thread
                    + ranks[i0] / block_size;
                input[i0] = values[i0];
                expected[i1] = values[i0];
            }
        }
    }

    // Preparing device
    type* device_input;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
    output_type* device_output;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
    unsigned int* device_ranks;
    CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_ranks, ranks.size() * sizeof(typename decltype(ranks)::value_type)));

    CUDA_CHECK(
        cudaMemcpy(
            device_input, input.data(),
            input.size() * sizeof(type),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            device_ranks, ranks.data(),
            ranks.size() * sizeof(unsigned int),
            cudaMemcpyHostToDevice
        )
    );

    // Running kernel
    constexpr unsigned int grid_size = (size / items_per_block);
    scatter_to_striped_kernel<type, output_type, items_per_block, items_per_thread>
        <<<dim3(grid_size), dim3(block_size), 0, 0>>> (
        device_input, device_output, device_ranks
    );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reading results
    CUDA_CHECK(
        cudaMemcpy(
            output.data(), device_output,
            output.size() * sizeof(typename decltype(output)::value_type),
            cudaMemcpyDeviceToHost
        )
    );

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(output[i], expected[i]);
    }

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_ranks));

}
