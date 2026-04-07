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
#include "cub/block/block_discontinuity.cuh"
#include "cub/thread/thread_operators.cuh"
#include "cub/block/block_load.cuh"
#include "cub/block/block_store.cuh"

template<
    class T,
    class Flag,
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class FlagOp
>
struct params
{
    using type = T;
    using flag_type = Flag;
    static constexpr unsigned int block_size = BlockSize;
    static constexpr unsigned int items_per_thread = ItemsPerThread;
    using flag_op_type = FlagOp;
};

template<class Params>
class HipcubBlockDiscontinuity : public ::testing::Test {
public:
    using params = Params;
};

template<class T>
struct custom_flag_op1
{
    HIPCUB_HOST_DEVICE
    bool operator()(const T& a, const T& b)
    {
        return (a == b);
    }
};

template<class T>
struct custom_flag_op2
{
    HIPCUB_HOST_DEVICE
    bool operator()(const T& a, const T& b) const
    {
        return (a - b > 5);
    }
};

template<class T, class FlagOp>
bool apply(FlagOp flag_op, const T& a, const T& b, unsigned int)
{
    return flag_op(a, b);
}

typedef ::testing::Types<
    // Power of 2 BlockSize
    params<unsigned int, int, 64U, 1, cub::Equality>,
    params<int, bool, 128U, 1, cub::Inequality>,
    params<float, int, 256U, 1, test_utils::less>,
    params<char, char, 1024U, 1, test_utils::less_equal>,
    params<int, bool, 256U, 1, custom_flag_op1<int> >,

    // Non-power of 2 BlockSize
    params<double, unsigned int, 65U, 1, test_utils::greater>,
    params<float, int, 37U, 1, custom_flag_op1<float> >,
    params<long long, char, 510U, 1, test_utils::greater_equal>,
    params<unsigned int, long long, 162U, 1, cub::Inequality>,
    params<unsigned char, bool, 255U, 1, cub::Equality>,

    // Power of 2 BlockSize and ItemsPerThread > 1
    params<int, char, 64U, 2, custom_flag_op2<int> >,
    params<int, short, 128U, 4, test_utils::less>,
    params<unsigned short, unsigned char, 256U, 7, custom_flag_op2<unsigned short> >,
    params<short, short, 512U, 8, cub::Equality>,

    // Non-power of 2 BlockSize and ItemsPerThread > 1
    params<double, int, 33U, 5, custom_flag_op2<double> >,
    params<double, unsigned int, 464U, 2, cub::Equality>,
    params<unsigned short, int, 100U, 3, test_utils::greater>,
    params<short, bool, 234U, 9, custom_flag_op1<short> >
> Params;

TYPED_TEST_SUITE(HipcubBlockDiscontinuity, Params);

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_heads_kernel(Type* device_input, long long* device_heads)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockDiscontinuity<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    if(blockIdx.x % 2 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.FlagHeads(head_flags, input, FlagOpType(), tile_predecessor_item);
    }
    else
    {
        bdiscontinuity.FlagHeads(head_flags, input, FlagOpType());
    }

    cub::StoreDirectBlocked(lid, device_heads + block_offset, head_flags);
}

TYPED_TEST(HipcubBlockDiscontinuity, FlagHeads)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    // std::vector<bool> is a special case that will cause an error in cudaMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, typename TestFixture::params::flag_type>::value,
                               int,
                               typename TestFixture::params::flag_type
                           >::type;
    using flag_type = typename TestFixture::params::flag_type;
    using flag_op_type = typename TestFixture::params::flag_op_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 2048;
    constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }


    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, type(0), type(10), seed_value);
        std::vector<long long> heads(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_heads(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = bi % 2 == 1
                        ? apply(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
                }
            }
        }

        // Preparing Device
        type* device_input;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_heads, heads.size() * sizeof(typename decltype(heads)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                cudaMemcpyHostToDevice
            )
        );

        // Running kernel
        flag_heads_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread>
            <<< dim3(grid_size), dim3(block_size), 0, 0>>> (            
            device_input, device_heads
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reading results
        CUDA_CHECK(
            cudaMemcpy(
                heads.data(), device_heads,
                heads.size() * sizeof(typename decltype(heads)::value_type),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(heads[i], expected_heads[i]);
        }

        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_heads));
    }
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_tails_kernel(Type* device_input, long long* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockDiscontinuity<Type, BlockSize> bdiscontinuity;

    FlagType tail_flags[ItemsPerThread];
    if(blockIdx.x % 2 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.FlagTails(tail_flags, input, FlagOpType(), tile_successor_item);
    }
    else
    {
        bdiscontinuity.FlagTails(tail_flags, input, FlagOpType());
    }

    cub::StoreDirectBlocked(lid, device_tails + block_offset, tail_flags);
}

TYPED_TEST(HipcubBlockDiscontinuity, FlagTails)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    // std::vector<bool> is a special case that will cause an error in cudaMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, typename TestFixture::params::flag_type>::value,
                               int,
                               typename TestFixture::params::flag_type
                           >::type;
    using flag_type = typename TestFixture::params::flag_type;
    using flag_op_type = typename TestFixture::params::flag_op_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 2048;
    constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }

    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, type(0), type(10), seed_value);
        std::vector<long long> tails(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = bi % 2 == 0
                        ? apply(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_tails;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_tails, tails.size() * sizeof(typename decltype(tails)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                cudaMemcpyHostToDevice
            )
        );

        // Running kernel
        flag_tails_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread >
            <<<dim3(grid_size), dim3(block_size), 0, 0 >>> (            
            device_input, device_tails
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reading results
        CUDA_CHECK(
            cudaMemcpy(
                tails.data(), device_tails,
                tails.size() * sizeof(typename decltype(tails)::value_type),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(tails[i], expected_tails[i]);
        }

        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_tails));
    }
}

template<
    class Type,
    class FlagType,
    class FlagOpType,
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__global__
__launch_bounds__(BlockSize)
void flag_heads_and_tails_kernel(Type* device_input, long long* device_heads, long long* device_tails)
{
    const unsigned int lid = threadIdx.x;
    const unsigned int items_per_block = BlockSize * ItemsPerThread;
    const unsigned int block_offset = blockIdx.x * items_per_block;

    Type input[ItemsPerThread];
    cub::LoadDirectBlocked(lid, device_input + block_offset, input);

    cub::BlockDiscontinuity<Type, BlockSize> bdiscontinuity;

    FlagType head_flags[ItemsPerThread];
    FlagType tail_flags[ItemsPerThread];
    if(blockIdx.x % 4 == 0)
    {
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.FlagHeadsAndTails(head_flags, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 1)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        const Type tile_successor_item = device_input[block_offset + items_per_block];
        bdiscontinuity.FlagHeadsAndTails(head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 2)
    {
        const Type tile_predecessor_item = device_input[block_offset - 1];
        bdiscontinuity.FlagHeadsAndTails(head_flags, tile_predecessor_item, tail_flags, input, FlagOpType());
    }
    else if(blockIdx.x % 4 == 3)
    {
        bdiscontinuity.FlagHeadsAndTails(head_flags, tail_flags, input, FlagOpType());
    }

    cub::StoreDirectBlocked(lid, device_heads + block_offset, head_flags);
    cub::StoreDirectBlocked(lid, device_tails + block_offset, tail_flags);
}

TYPED_TEST(HipcubBlockDiscontinuity, FlagHeadsAndTails)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    CUDA_CHECK(cudaSetDevice(device_id));

    using type = typename TestFixture::params::type;
    // std::vector<bool> is a special case that will cause an error in cudaMemcpy
    using stored_flag_type = typename std::conditional<
                               std::is_same<bool, typename TestFixture::params::flag_type>::value,
                               int,
                               typename TestFixture::params::flag_type
                           >::type;
    using flag_type = typename TestFixture::params::flag_type;
    using flag_op_type = typename TestFixture::params::flag_op_type;
    constexpr size_t block_size = TestFixture::params::block_size;
    constexpr size_t items_per_thread = TestFixture::params::items_per_thread;
    constexpr size_t items_per_block = block_size * items_per_thread;
    const size_t size = items_per_block * 2048;
    constexpr size_t grid_size = size / items_per_block;

    // Given block size not supported
    if(block_size > test_utils::get_max_block_size())
    {
        return;
    }


    for (size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        unsigned int seed_value = seed_index < random_seeds_count  ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

        // Generate data
        std::vector<type> input = test_utils::get_random_data<type>(size, type(0), type(10), seed_value);
        std::vector<long long> heads(size);
        std::vector<long long> tails(size);

        // Calculate expected results on host
        std::vector<stored_flag_type> expected_heads(size);
        std::vector<stored_flag_type> expected_tails(size);
        flag_op_type flag_op;
        for(size_t bi = 0; bi < size / items_per_block; bi++)
        {
            for(size_t ii = 0; ii < items_per_block; ii++)
            {
                const size_t i = bi * items_per_block + ii;
                if(ii == 0)
                {
                    expected_heads[i] = (bi % 4 == 1 || bi % 4 == 2)
                        ? apply(flag_op, input[i - 1], input[i], ii)
                        : flag_type(true);
                }
                else
                {
                    expected_heads[i] = apply(flag_op, input[i - 1], input[i], ii);
                }
                if(ii == items_per_block - 1)
                {
                    expected_tails[i] = (bi % 4 == 0 || bi % 4 == 1)
                        ? apply(flag_op, input[i], input[i + 1], ii + 1)
                        : flag_type(true);
                }
                else
                {
                    expected_tails[i] = apply(flag_op, input[i], input[i + 1], ii + 1);
                }
            }
        }

        // Preparing Device
        type* device_input;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_input, input.size() * sizeof(typename decltype(input)::value_type)));
        long long* device_heads;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_heads, tails.size() * sizeof(typename decltype(heads)::value_type)));
        long long* device_tails;
        CUDA_CHECK(test_common_utils::cudaMallocHelper(&device_tails, tails.size() * sizeof(typename decltype(tails)::value_type)));

        CUDA_CHECK(
            cudaMemcpy(
                device_input, input.data(),
                input.size() * sizeof(type),
                cudaMemcpyHostToDevice
            )
        );

        // Running kernel
        flag_heads_and_tails_kernel<
                    type, flag_type, flag_op_type,
                    block_size, items_per_thread>
            <<<dim3(grid_size), dim3(block_size), 0, 0>>> ( 
            device_input, device_heads, device_tails
        );
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reading results
        CUDA_CHECK(
            cudaMemcpy(
                heads.data(), device_heads,
                heads.size() * sizeof(typename decltype(heads)::value_type),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                tails.data(), device_tails,
                tails.size() * sizeof(typename decltype(tails)::value_type),
                cudaMemcpyDeviceToHost
            )
        );

        // Validating results
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(heads[i], expected_heads[i]);
            ASSERT_EQ(tails[i], expected_tails[i]);
        }

        CUDA_CHECK(cudaFree(device_input));
        CUDA_CHECK(cudaFree(device_heads));
        CUDA_CHECK(cudaFree(device_tails));
    }
}
