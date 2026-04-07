# MIT License #
# Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Gets a test target name based on the first source file.
function(get_hipcub_test_target TEST_SOURCES TEST_TARGET)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  set(TEST_TARGET ${TEST_TARGET} PARENT_SCOPE)
endfunction()

function(add_hipcub_test TEST_NAME TEST_SOURCES)
  get_hipcub_test_target(${TEST_SOURCES} TEST_TARGET)
  add_hipcub_test_internal(${TEST_NAME} "${TEST_SOURCES}" ${TEST_TARGET})
endfunction()

function(add_hipcub_test_internal TEST_NAME TEST_SOURCES TEST_TARGET)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_include_directories(${TEST_TARGET} SYSTEM BEFORE
    PUBLIC
        $<BUILD_INTERFACE:${GTEST_SRC}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  )
  target_link_libraries(${TEST_TARGET}
    PRIVATE
       gtest
       gtest_main
  )

  set_target_properties(${TEST_TARGET}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test/cub"
  )
  add_relative_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# Returns a list of values that match the pattern
# "if ${WORD} == <value>" with optional whitespace between the tokens.
function(get_match_list FILE_STRING WORD MATCH_LIST)
  # collect all substrings that match the pattern
  string(REGEX MATCHALL "${WORD}[ ]*==[ ]*[0-9]*" LINE_MATCHES "${${FILE_STRING}}")
  set(${MATCH_LIST} "")
  # iterate over the substrings, record the values using the same regex
  foreach(LINE IN LISTS LINE_MATCHES)
    string(REGEX MATCH "${WORD}[ ]*==[ ]*([0-9]*)" TMP "${LINE}")
    list(APPEND ${MATCH_LIST} "${CMAKE_MATCH_1}")
  endforeach()
  set(${MATCH_LIST} ${${MATCH_LIST}} PARENT_SCOPE)
endfunction()

function(add_hipcub_test_parallel TEST_NAME TEST_SOURCE)
  get_hipcub_test_target(${TEST_SOURCE} TEST_TARGET)

  file(READ ${TEST_SOURCE} FILE_CONTENTS)

  set(SOURCES "")

  # first, handle all non-typed tests
  # disable typed test, generate one file for each non-typed test
  set(HIPCUB_TEST_SUITE_SLICE -1)
  set(HIPCUB_TEST_TYPE_SLICE -1)
  get_match_list(FILE_CONTENTS "HIPCUB_TEST_SLICE" TEST_SLICE_LIST)
  list(LENGTH TEST_SLICE_LIST TEST_SLICE_COUNT)
  if(TEST_SLICE_COUNT EQUAL 0)
    message(VERBOSE "found no non-typed tests for test target ${TEST_TARGET}")
  else()
    message(VERBOSE "found ${TEST_SLICE_COUNT} non-typed test slice(s) for test target ${TEST_TARGET}")
    foreach(HIPCUB_TEST_SLICE IN LISTS TEST_SLICE_LIST)
      set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_${HIPCUB_TEST_SLICE}.cpp")
      configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
      list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
    endforeach()
  endif()

  # second, handle all typed tests
  # disable non-typed test, generate one file for each test suite and test type pair
  set(HIPCUB_TEST_SLICE -1)
  get_match_list(FILE_CONTENTS "HIPCUB_TEST_SUITE_SLICE" TEST_SUITE_SLICE_LIST)
  list(LENGTH TEST_SUITE_SLICE_LIST TEST_SUITE_SLICE_COUNT)
  get_match_list(FILE_CONTENTS "HIPCUB_TEST_TYPE_SLICE" TEST_TYPE_SLICE_LIST)
  list(LENGTH TEST_TYPE_SLICE_LIST TEST_TYPE_SLICE_COUNT)
  if(TEST_SUITE_SLICE_COUNT EQUAL 0 OR TEST_TYPE_SLICE_COUNT EQUAL 0)
    message(VERBOSE "found no typed tests for test target ${TEST_TARGET}")
  else()
    message(VERBOSE "found ${TEST_SUITE_SLICE_COUNT} test suite slice(s) and \
${TEST_TYPE_SLICE_COUNT} test type slice(s) for test target ${TEST_TARGET}")
    foreach(HIPCUB_TEST_SUITE_SLICE IN LISTS TEST_SUITE_SLICE_LIST)
      foreach(HIPCUB_TEST_TYPE_SLICE IN LISTS TEST_TYPE_SLICE_LIST)
        set(FILENAME "${TEST_TARGET}.parallel/${TEST_TARGET}_typed_${HIPCUB_TEST_SUITE_SLICE}_${HIPCUB_TEST_TYPE_SLICE}.cpp")
        configure_file(${TEST_SOURCE} ${FILENAME} @ONLY)
        list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}")
      endforeach()
    endforeach()
  endif()

  # if no files are generated, nothing is built for the target
  list(LENGTH SOURCES SOURCES_COUNT)
  if(${SOURCES_COUNT} EQUAL 0)
    message(FATAL_ERROR "no .cpp files generated for target ${TEST_TARGET}")
  endif()

  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${TEST_TARGET}.parallel")
  add_hipcub_test_internal(${TEST_NAME} "${SOURCES}" ${TEST_TARGET})
  target_include_directories("${TEST_TARGET}" PRIVATE "../../test/cub")
endfunction()

# ****************************************************************************
# Tests
# ****************************************************************************

# HIP basic test, which also checks if there are no linkage problems when there are multiple sources
add_hipcub_test("cub.BasicTest" "test_hipcub_basic.cpp;detail/get_hipcub_version.cpp")

add_hipcub_test("cub.CachingDeviceAllocator" test_hipcub_caching_device_allocator.cpp)
add_hipcub_test("cub.BlockAdjacentDifference" test_hipcub_block_adjacent_difference.cpp)
add_hipcub_test("cub.BlockDiscontinuity" test_hipcub_block_discontinuity.cpp)
add_hipcub_test("cub.BlockExchange" test_hipcub_block_exchange.cpp)
add_hipcub_test("cub.BlockHistogram" test_hipcub_block_histogram.cpp)
add_hipcub_test("cub.BlockLoadStore" test_hipcub_block_load_store.cpp)
add_hipcub_test("cub.BlockMergeSort" test_hipcub_block_merge_sort.cpp)
add_hipcub_test("cub.BlockRadixRank" test_hipcub_block_radix_rank.cpp)
add_hipcub_test("cub.BlockRadixSort" test_hipcub_block_radix_sort.cpp)
add_hipcub_test("cub.BlockReduce" test_hipcub_block_reduce.cpp)
add_hipcub_test("cub.BlockRunLengthDecode" test_hipcub_block_run_length_decode.cpp)
add_hipcub_test("cub.BlockScan" test_hipcub_block_scan.cpp)
add_hipcub_test("cub.BlockShuffle" test_hipcub_block_shuffle.cpp)
#add_hipcub_test("cub.DeviceAdjacentDifference" test_hipcub_device_adjacent_difference.cpp)
add_hipcub_test("cub.DeviceHistogram" test_hipcub_device_histogram.cpp)
add_hipcub_test("cub.DeviceMergeSort" test_hipcub_device_merge_sort.cpp)
add_hipcub_test_parallel("cub.DeviceRadixSort" test_hipcub_device_radix_sort.cpp.in)
add_hipcub_test("cub.DeviceReduce" test_hipcub_device_reduce.cpp)
add_hipcub_test("cub.DeviceRunLengthEncode" test_hipcub_device_run_length_encode.cpp)
add_hipcub_test("cub.DeviceReduceByKey" test_hipcub_device_reduce_by_key.cpp)
add_hipcub_test("cub.DeviceScan" test_hipcub_device_scan.cpp)
add_hipcub_test_parallel("cub.DeviceSegmentedRadixSort" test_hipcub_device_segmented_radix_sort.cpp.in)
add_hipcub_test("cub.DeviceSegmentedReduce" test_hipcub_device_segmented_reduce.cpp)
add_hipcub_test_parallel("cub.DeviceSegmentedSort" test_hipcub_device_segmented_sort.cpp.in)
add_hipcub_test("cub.DeviceSelect" test_hipcub_device_select.cpp)
add_hipcub_test("cub.DevicePartition" test_hipcub_device_partition.cpp)
add_hipcub_test("cub.Grid" test_hipcub_grid.cpp)
add_hipcub_test("cub.UtilPtx" test_hipcub_util_ptx.cpp)
add_hipcub_test("cub.WarpExchange" test_hipcub_warp_exchange.cpp)
add_hipcub_test("cub.WarpLoad" test_hipcub_warp_load.cpp)
add_hipcub_test("cub.WarpMergeSort" test_hipcub_warp_merge_sort.cpp)
add_hipcub_test("cub.WarpReduce" test_hipcub_warp_reduce.cpp)
add_hipcub_test("cub.WarpScan" test_hipcub_warp_scan.cpp)
add_hipcub_test("cub.WarpStore" test_hipcub_warp_store.cpp)
add_hipcub_test("cub.Iterator" test_hipcub_iterators.cpp)
add_hipcub_test("cub.ThreadOperations" test_hipcub_thread.cpp)
add_hipcub_test("cub.ThreadSort" test_hipcub_thread_sort.cpp)
