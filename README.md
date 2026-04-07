## What is this repository for? ###

The project allows developers to test rocm hip API.

## Compiler ###
The project default compiler path is /opt/rocm/bin/hipcc, rocm hip default path is /opt/rocm/, if you do not need to change compiler and rocm path, just cmake.such as:
```shell
mkdir build
cd build
cmake ..
make
```
if developers want to use debug version, just like this:
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```
### How to compiler with hiphsa? ####

if developers want to compiler the project with hiphsa, need to use cmake -D command, add some variable:
* TEST_PLATFORM: rocm or hiphsa
* HIP_INCLUDE_PATH: hip include path
* HIP_LIB_PATH: hip lib path
* TEST_PLATFORM: test platform galaxy or rocm

Before cmake, if developer does not want to use default compiler, need change HIP_CLANG_PATH env var, such as:
```shell
export HIP_CLANG_PATH=/data_share/llvm-project/build/bin
mkdir build
cd build
cmake -DTEST_PLATFORM=hiphsa -DHIP_INCLUDE_PATH=/data_share/buildtest/hzk/hiphsa/include -DHIP_LIB_PATH=/data_share/buildtest/hzk/hiphsa/build -DCMAKE_BUILD_TYPE=Debug -DTEST_PLATFORM=galaxy..
make
```

## How to run case? ###
if developers want to run all case, just run hiptest
```shell
./hiptest
```
if developers want to run one case ,for example just want to run hipHostRegisterTest3 case:
```shell
./hip_test --gtest_filter=hipMemory.hipHostRegisterTest3
```
if developer want to run all module case, for example just want to run all hiMemory case:
```shell
./hip_test --gtest_filter=hipMemory.*
```

## How to add test moudle? ###
For example，we want to add device moudle. In the file runtimeAPI.cmake, load and run the cmake code from the module stream. The file runtimeAPI.cmake path is /hiptest/src/runtimeAPI/runtimeAPI.cmake:
```shell
set(RUMTIMEAPI_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/runtimeAPI)

message(STATUS "include runtime api")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
    ${RUMTIMEAPI_ROOT_PATH}/device #add device moudle path
    ${RUMTIMEAPI_ROOT_PATH}/stream  
    ${RUMTIMEAPI_ROOT_PATH}/memory
    ${RUMTIMEAPI_ROOT_PATH}/event
) 
include(device) #Load and run CMake code from device module.
include(stream)	
include(memory)
include(event)
set(RUNTIMEAPI_TEST_FILES 
    ${RUNTIMEAPI_DEVICE_TEST_FILES} #run
    ${RUNTIMEAPI_STREAM_TEST_FILES} 
    ${RUNTIMEAPI_MEMORY_TEST_FILES}
    ${RUNTIMEAPI_EVENT_TEST_FILES}
)

#message(STATUS "RUNTIMEAPI_TEST_FILES: ${RUNTIMEAPI_TEST_FILES}")
```
Next, add the device folder. In the device folder, add the device module's cmake code and test case.
```shell
├── runtimeAPI
│   ├── device
│   │   ├── hipGetDeviceTest.cpp 
│   │   ├── hipGetDeviceCountTest.cpp
│   │   └── device.cmake 
│   ├── stream
│   ├── evet  
│   └── runtimeAPI.cmake
```
Device module's cmake code filename **device**.cmake should be the same as the include(**device**) in runtimeAPI.cmake.

In device. cmake, add the test case path:
```shell
message(STATUS "include device test case")
set(RUNTIMEAPI_DEVICE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtimeAPI/device)
set(RUNTIMEAPI_DEVICE_TEST_FILES
${RUNTIMEAPI_DEVICE_DIR}/hipGetDeviceTest.cpp
        ${RUNTIMEAPI_DEVICE_DIR}/hipSetDeviceTest.cpp
        ${RUNTIMEAPI_DEVICE_DIR}/hipGetDeviceTest.cpp #add hipGetDevice testcase
        ${RUNTIMEAPI_DEVICE_DIR}/hipGetDeviceCountTest.cpp
        ${RUNTIMEAPI_DEVICE_DIR}/hipSetDeviceFlagsTest.cpp
  )
```

## How to write test case? ###
#### Define test function
When writing unit tests using GoogleTest, use the TEST() macro to define and name a test function. 
```shell
TEST(TestSuiteName, TestName) {
  ... test body ...
}
```
TEST() arguments go from general to specific. The first argument is the name of the test suite, and the second argument is the test's name within the test suite.For example, the test path is /hiptest/src/runtimeAPI/device/hipGetDeviceTest.cpp:
```shell
TEST(hipDevice, hipGetDeviceTest)
{
    int deviceID = 0;
    hipError_t ret = hipSuccess;
    ret= hipGetDevice(&deviceID);
    EXPECT_EQ(ret, hipSuccess);
    EXPECT_EQ(deviceID, 0);
}
```
If the test module is **Device**, the name of the test suite is **hipDevice**. If the test API function is **hipGetDevice**, the test's name is **hipGetDeviceTest**. And then the filename is **hipGetDeviceTest.cpp**.

#### Assertions
Googletest assertions are macros that resemble function calls. You test a class or function by making assertions about its behavior.The assertions come in pairs that test the same thing but have different effects on the current function.

#### Basic Assertions
These assertions do basic true/false condition testing.

Fatal assertion            | Nonfatal assertion         | Verifies
-------------------------- | -------------------------- | --------------------
`ASSERT_TRUE(condition);`  | `EXPECT_TRUE(condition);`  | `condition` is true
`ASSERT_FALSE(condition);` | `EXPECT_FALSE(condition);` | `condition` is false

`ASSERT_*` versions generate fatal failures when they fail, and **abort the current function**.
`EXPECT_*` versions generate nonfatal failures, which don't abort the current function. Usually `EXPECT_*` are preferred, as they allow more than one failure to be reported in a test.
However, you should use `ASSERT_*` if it doesn't make sense to continue when the assertion in question fails.

#### Binary Comparison
This section describes assertions that compare two values.

Fatal assertion          | Nonfatal assertion       | Verifies
------------------------ | ------------------------ | --------------
`ASSERT_EQ(val1, val2);` | `EXPECT_EQ(val1, val2);` | `val1 == val2`
`ASSERT_NE(val1, val2);` | `EXPECT_NE(val1, val2);` | `val1 != val2`
`ASSERT_LT(val1, val2);` | `EXPECT_LT(val1, val2);` | `val1 < val2`
`ASSERT_LE(val1, val2);` | `EXPECT_LE(val1, val2);` | `val1 <= val2`
`ASSERT_GT(val1, val2);` | `EXPECT_GT(val1, val2);` | `val1 > val2`
`ASSERT_GE(val1, val2);` | `EXPECT_GE(val1, val2);` | `val1 >= val2`

#### String Comparison
The assertions in this group compare two **C strings**. If you want to compare two `string` objects, use `EXPECT_EQ`, `EXPECT_NE`, and etc instead.

<!-- mdformat off(github rendering does not support multiline tables) -->

| Fatal assertion                | Nonfatal assertion             | Verifies                                                 |
| --------------------------     | ------------------------------ | -------------------------------------------------------- |
| `ASSERT_STREQ(str1,str2);`     | `EXPECT_STREQ(str1,str2);`     | the two C strings have the same content   		     |
| `ASSERT_STRNE(str1,str2);`     | `EXPECT_STRNE(str1,str2);`     | the two C strings have different contents 		     |
| `ASSERT_STRCASEEQ(str1,str2);` | `EXPECT_STRCASEEQ(str1,str2);` | the two C strings have the same content, ignoring case   |
| `ASSERT_STRCASENE(str1,str2);` | `EXPECT_STRCASENE(str1,str2);` | the two C strings have different contents, ignoring case |

<!-- mdformat on-->

Note that "CASE" in an assertion name means that case is ignored. A `NULL` pointer and an empty string are considered *different*.

`*STREQ*` and `*STRNE*` also accept wide C strings (`wchar_t*`). If a comparison of two wide strings fails, their values will be printed as UTF-8 narrow strings.

## How to collect test results?

For testing the performance of Galaxy, each test case needed to be run more than ten times to get the average of time taken. The reason for using the average time is to avoid the jitter brought by external factors.

To simplify the work, `performance.py` is written to control the number of test cases to run, search key information, and calculate the average time. The test result will be printed in terminal.

The command of `performance.py` :

```
python3 performance.py "path of hip_test" "name of test set"
```

Example :

```
python3 performance.py ../build/hip_test hipPerformancehipMemset
```

The goal of `multi_performance_test.py` is to test multiple sets of test cases and organize the data to text file, which will help us for the further operation.

The command of `multi_performance_test.py` :

```
python3 multi_performance_test.py "path of hip_test" 
```

Example :

```
python3 multi_performance_test.py ../build/hip_test 
```

The list named "testsuite" can be modified to control the sets of test cases:

```
testsuite=['hipPerformanceUnpinD2H','hipPerformanceUnpinH2D']
```

