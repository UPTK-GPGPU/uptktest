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
#include "cub/block/block_scan.cuh"

// Params for tests
template<
    class T,
    unsigned int BlockSize = 256U,
    unsigned int ItemsPerThread = 1U,
    cub::BlockScanAlgorithm Algorithm = cub::BLOCK_SCAN_WARP_SCANS
>
struct params
{
    using type = T;
    static constexpr cub::BlockScanAlgorithm algorithm = Algorithm;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubBlockScanSingleValueTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr cub::BlockScanAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int block_size = Params::block_size;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // cub::BLOCK_SCAN_WARP_SCANS
    // -----------------------------------------------------------------------
    params<int, 64U>,
    params<int, 128U>,
    params<int, 256U>,
    params<int, 512U>,
    params<int, 65U>,
    params<int, 37U>,
    params<int, 162U>,
    params<int, 255U>,
    // uint tests
    params<unsigned int, 64U>,
    params<unsigned int, 256U>,
    params<unsigned int, 377U>,
    // long tests
    params<long, 64U>,
    params<long, 256U>,
    params<long, 377U>,
    // -----------------------------------------------------------------------
    // cub::BLOCK_SCAN_RAKING
    // -----------------------------------------------------------------------
    params<int, 64U, 1, cub::BLOCK_SCAN_RAKING>,
    params<int, 128U, 1, cub::BLOCK_SCAN_RAKING>,
    params<int, 256U, 1, cub::BLOCK_SCAN_RAKING>,
    params<int, 512U, 1, cub::BLOCK_SCAN_RAKING>,
    params<unsigned long, 65U, 1, cub::BLOCK_SCAN_RAKING>,
    params<long, 37U, 1, cub::BLOCK_SCAN_RAKING>,
    params<short, 162U, 1, cub::BLOCK_SCAN_RAKING>,
    params<unsigned int, 255U, 1, cub::BLOCK_SCAN_RAKING>,
    params<int, 377U, 1, cub::BLOCK_SCAN_RAKING>,
    params<unsigned char, 377U, 1, cub::BLOCK_SCAN_RAKING>
> SingleValueTestParams;

TYPED_TEST_SUITE(HipcubBlockScanSingleValueTests, SingleValueTestParams);

template<
    unsigned int BlockSize,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_kernel(T* device_output)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(value, value, cub::Sum());

    device_output[index] = value;
}

TYPED_TEST(HipcubBlockScanSingleValueTests, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        inclusive_scan_kernel<block_size, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_reduce_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    T reduction;
    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(value, value, cub::Sum(), reduction);
    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, InclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_reductions[i] = expected[(i+1) * block_size - 1];
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        inclusive_scan_reduce_kernel<block_size, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>> (
            device_output, device_output_reductions
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_reductions));
    }
}

template<
    unsigned int BlockSize,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(value, value, cub::Sum(), prefix_callback);

    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_bp[blockIdx.x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, InclusiveScanPrefixCallback)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / block_size);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            expected[i * block_size] = block_prefix;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_block_prefixes[i] = expected[(i+1) * block_size - 1];
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        inclusive_scan_prefix_callback_kernel<block_size, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_bp, block_prefix
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_EQ(output_block_prefixes[i], expected_block_prefixes[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_bp));
    }
}

template<
    unsigned int BlockSize,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void exclusive_scan_kernel(T* device_output, T init)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(value, value, init, cub::Sum());
    device_output[index] = value;
}

TYPED_TEST(HipcubBlockScanSingleValueTests, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 241, seed_value);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
          expected[i * block_size] = init;
          for(size_t j = 1; j < block_size; j++)
          {
              auto idx = i * block_size + j;
              expected[idx] = output[idx-1] + expected[idx-1];
          }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        CUDA_CHECK(
          cudaMemcpy(
              device_output, output.data(),
              output.size() * sizeof(T),
              cudaMemcpyHostToDevice
          )
        );

        // Launching kernel
        exclusive_scan_kernel<block_size, algorithm, T>
          <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
          device_output, init
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
          cudaMemcpy(
              output.data(), device_output,
              output.size() * sizeof(T),
              cudaMemcpyDeviceToHost
          )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
          ASSERT_EQ(output[i], expected[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void exclusive_scan_reduce_kernel(T* device_output, T* device_output_reductions, T init)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T value = device_output[index];
    T reduction;
    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(value, value, init, cub::Sum(), reduction);
    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, ExclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            expected[i * block_size] = init;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }

            expected_reductions[i] = 0;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected_reductions[i] += output[idx];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        exclusive_scan_reduce_kernel<block_size, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_reductions, init
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_EQ(output_reductions[i], expected_reductions[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_reductions));
    }
}

template<
    unsigned int BlockSize,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void exclusive_scan_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = (blockIdx.x * BlockSize) + threadIdx.x;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    T value = device_output[index];

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(value, value, cub::Sum(), prefix_callback);

    device_output[index] = value;
    if(threadIdx.x == 0)
    {
        device_output_bp[blockIdx.x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, ExclusiveScanPrefixCallback)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / block_size);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            expected[i * block_size] = block_prefix;
            for(size_t j = 1; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }

            expected_block_prefixes[i] = block_prefix;
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected_block_prefixes[i] += output[idx];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        exclusive_scan_prefix_callback_kernel<block_size, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_bp, block_prefix
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_EQ(output_block_prefixes[i], expected_block_prefixes[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_bp));
    }
}

TYPED_TEST(HipcubBlockScanSingleValueTests, CustomStruct)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using base_type = typename TestFixture::type;
    using T = test_utils::custom_test_type<base_type>;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t size = block_size * 113;
    const size_t grid_size = size / block_size;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output(size);
        {
            std::vector<base_type> random_values =
                test_utils::get_random_data<base_type>(2 * output.size(), 2, 200, seed_value);
            for(size_t i = 0; i < output.size(); i++)
            {
                output[i].x = random_values[i],
                output[i].y = random_values[i + output.size()];
            }
        }

        // Calculate expected results on host
        std::vector<T> expected(output.size(), T(0));
        for(size_t i = 0; i < output.size() / block_size; i++)
        {
            for(size_t j = 0; j < block_size; j++)
            {
                auto idx = i * block_size + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        inclusive_scan_kernel<block_size, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_EQ(output[i], expected[i]);
        }

        CUDA_CHECK(cudaFree(device_output));
    }
}

// // ---------------------------------------------------------
// // Test for scan ops taking array of values as input
// // ---------------------------------------------------------

template<class Params>
class HipcubBlockScanInputArrayTests : public ::testing::Test
{
public:
    using type = typename Params::type;
    static constexpr unsigned int block_size = Params::block_size;
    static constexpr cub::BlockScanAlgorithm algorithm = Params::algorithm;
    static constexpr unsigned int items_per_thread = Params::items_per_thread;
};

typedef ::testing::Types<
    // -----------------------------------------------------------------------
    // cub::BlockScanAlgorithm::using_warp_scan
    // -----------------------------------------------------------------------
    params<float, 6U,   32>,
    params<float, 32,   2>,
    params<unsigned int, 256,  3>,
    params<int, 512,  4>,
    params<float, 37,   2>,
    params<float, 65,   5>,
    params<float, 162,  7>,
    params<float, 255,  15>,
    // -----------------------------------------------------------------------
    // cub::BLOCK_SCAN_RAKING
    // -----------------------------------------------------------------------
    params<float, 6U,   32, cub::BLOCK_SCAN_RAKING>,
    params<float, 32,   2,  cub::BLOCK_SCAN_RAKING>,
    params<int, 256,  3,  cub::BLOCK_SCAN_RAKING>,
    params<unsigned int, 512,  4,  cub::BLOCK_SCAN_RAKING>,
    params<float, 37,   2,  cub::BLOCK_SCAN_RAKING>,
    params<float, 65,   5,  cub::BLOCK_SCAN_RAKING>,
    params<float, 162,  7,  cub::BLOCK_SCAN_RAKING>,
    params<float, 255,  15, cub::BLOCK_SCAN_RAKING>
> InputArrayTestParams;

TYPED_TEST_SUITE(HipcubBlockScanInputArrayTests, InputArrayTestParams);

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_array_kernel(T* device_output)
{
    const unsigned int index = ((blockIdx.x * BlockSize ) + threadIdx.x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(in_out, in_out, cub::Sum());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

}

TYPED_TEST(HipcubBlockScanInputArrayTests, InclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        inclusive_scan_array_kernel<block_size, items_per_thread, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        CUDA_CHECK(cudaFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions)
{
    const unsigned int index = ((blockIdx.x * BlockSize ) + threadIdx.x) * ItemsPerThread;

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    T reduction;
    bscan_t(temp_storage).InclusiveScan(in_out, in_out, cub::Sum(), reduction);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, InclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_reductions[i] = expected[(i+1) * items_per_block - 1];
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        inclusive_scan_reduce_array_kernel<block_size, items_per_thread, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_reductions
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_NEAR(
                output_reductions[i], expected_reductions[i],
                static_cast<T>(0.05) * expected_reductions[i]
            );
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_reductions));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void inclusive_scan_array_prefix_callback_kernel(T* device_output, T* device_output_bp, T block_prefix)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).InclusiveScan(in_out, in_out, cub::Sum(), prefix_callback);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_bp[blockIdx.x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, InclusiveScanPrefixCallback)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / items_per_block, 0);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = block_prefix;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx] + expected[j > 0 ? idx-1 : idx];
            }
            expected_block_prefixes[i] = expected[(i+1) * items_per_block - 1];
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output_bp, output_block_prefixes.data(),
                output_block_prefixes.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        
                inclusive_scan_array_prefix_callback_kernel<block_size, items_per_thread, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_bp, block_prefix
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_NEAR(
                output_block_prefixes[i], expected_block_prefixes[i],
                static_cast<T>(0.05) * expected_block_prefixes[i]
            );
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_bp));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void exclusive_scan_array_kernel(T* device_output, T init)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(in_out, in_out, init, cub::Sum());

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, ExclusiveScan)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = init;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        exclusive_scan_array_kernel<block_size, items_per_thread, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, init
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        CUDA_CHECK(cudaFree(device_output));
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void exclusive_scan_reduce_array_kernel(T* device_output, T* device_output_reductions, T init)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index + j];
    }

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    T reduction;
    bscan_t(temp_storage).ExclusiveScan(in_out, in_out, init, cub::Sum(), reduction);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_reductions[blockIdx.x] = reduction;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, ExclusiveScanReduce)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);

        // Output reduce results
        std::vector<T> output_reductions(size / block_size, 0);
        const T init = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_reductions(output_reductions.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = init;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }
            for(size_t j = 0; j < items_per_block; j++)
            {
                expected_reductions[i] += output[i * items_per_block + j];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_reductions;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_reductions,
                  output_reductions.size() * sizeof(typename decltype(output_reductions)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemset(
                device_output_reductions, T(0), output_reductions.size() * sizeof(T)
            )
        );

        // Launching kernel
        
                exclusive_scan_reduce_array_kernel<block_size, items_per_thread, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_reductions, init
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_reductions.data(), device_output_reductions,
                output_reductions.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_reductions.size(); i++)
        {
            ASSERT_NEAR(
                output_reductions[i], expected_reductions[i],
                static_cast<T>(0.05) * expected_reductions[i]
            );
        }
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    cub::BlockScanAlgorithm Algorithm,
    class T
>
__global__
__launch_bounds__(BlockSize)
void exclusive_scan_prefix_callback_array_kernel(
    T* device_output,
    T* device_output_bp,
    T block_prefix
)
{
    const unsigned int index = ((blockIdx.x * BlockSize) + threadIdx.x) * ItemsPerThread;
    T prefix_value = block_prefix;
    auto prefix_callback = [&prefix_value](T reduction)
    {
        T prefix = prefix_value;
        prefix_value += reduction;
        return prefix;
    };

    // load
    T in_out[ItemsPerThread];
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        in_out[j] = device_output[index+ j];
    }

    using bscan_t = cub::BlockScan<T, BlockSize, Algorithm>;
    __shared__ typename bscan_t::TempStorage temp_storage;
    bscan_t(temp_storage).ExclusiveScan(in_out, in_out, cub::Sum(), prefix_callback);

    // store
    for(unsigned int j = 0; j < ItemsPerThread; j++)
    {
        device_output[index + j] = in_out[j];
    }

    if(threadIdx.x == 0)
    {
        device_output_bp[blockIdx.x] = prefix_value;
    }
}

TYPED_TEST(HipcubBlockScanInputArrayTests, ExclusiveScanPrefixCallback)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using T = typename TestFixture::type;
    constexpr auto algorithm = TestFixture::algorithm;
    constexpr size_t block_size = TestFixture::block_size;
    constexpr size_t items_per_thread = TestFixture::items_per_thread;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    const size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 37;
    const size_t grid_size = size / items_per_block;

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<T> output = test_utils::get_random_data<T>(size, 2, 200, seed_value);
        std::vector<T> output_block_prefixes(size / items_per_block);
        T block_prefix = test_utils::get_random_value<T>(
            0,
            100,
            seed_value + seed_value_addition
        );

        // Calculate expected results on host
        std::vector<T> expected(output.size(), 0);
        std::vector<T> expected_block_prefixes(output_block_prefixes.size(), 0);
        for(size_t i = 0; i < output.size() / items_per_block; i++)
        {
            expected[i * items_per_block] = block_prefix;
            for(size_t j = 1; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected[idx] = output[idx-1] + expected[idx-1];
            }
            expected_block_prefixes[i] = block_prefix;
            for(size_t j = 0; j < items_per_block; j++)
            {
                auto idx = i * items_per_block + j;
                expected_block_prefixes[i] += output[idx];
            }
        }

        // Writing to device memory
        T* device_output;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_output, output.size() * sizeof(typename decltype(output)::value_type)));
        T* device_output_bp;
        CUDA_CHECK(
            test_common_utils::cudaMallocHelper(
                  &device_output_bp,
                  output_block_prefixes.size() * sizeof(typename decltype(output_block_prefixes)::value_type)
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                device_output, output.data(),
                output.size() * sizeof(T),
                cudaMemcpyHostToDevice
            )
        );

        // Launching kernel
        
                exclusive_scan_prefix_callback_array_kernel<block_size, items_per_thread, algorithm, T>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>>(
            device_output, device_output_bp, block_prefix
        );

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read from device memory
        CUDA_CHECK(
            cudaMemcpy(
                output.data(), device_output,
                output.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                output_block_prefixes.data(), device_output_bp,
                output_block_prefixes.size() * sizeof(T),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < output.size(); i++)
        {
            ASSERT_NEAR(
                output[i], expected[i],
                static_cast<T>(0.05) * expected[i]
            );
        }

        for(size_t i = 0; i < output_block_prefixes.size(); i++)
        {
            ASSERT_NEAR(
                output_block_prefixes[i], expected_block_prefixes[i],
                static_cast<T>(0.05) * expected_block_prefixes[i]
            );
        }

        CUDA_CHECK(cudaFree(device_output));
        CUDA_CHECK(cudaFree(device_output_bp));
    }
}
