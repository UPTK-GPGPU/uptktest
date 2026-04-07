#include <stdio.h>
#include <string.h>
#include <gtest/gtest.h>
#include <gtest/test_common.h>

#include <nvrtc.h>
#include <cuda.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

static constexpr auto program{
    R"(
extern "C"
__global__ void kernel(int* a) {
  // C++17 feature
  if (int j = 10; *a % 2 == 0)
    *a = 10 + j;
  else
    *a = 20 + j;
}
)"};

void test1(){
  using namespace std;
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog,         // prog
                      program,       // buffer
                      "program.cu",  // name
                      0, nullptr, nullptr);
  UPTKDeviceProp props;
  int device = 0;
  CUDACHECK(UPTKGetDeviceProperties(&props, device));

    string sarg = string("--gpu-architecture=compute_")
    + to_string(props.major) + to_string(props.minor);

  const char* options[] = {sarg.c_str(), "-std=c++17", "-Werror"};
  nvrtcResult compileResult{nvrtcCompileProgram(prog, 3, options)};
  size_t logSize;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  nvrtcDestroyProgram(&prog);
  EXPECT_EQ(compileResult, NVRTC_SUCCESS);
}

static constexpr const char template_kernel[]{R"(
template <typename T> struct complex {
 public:
  typedef T value_type;
  inline __host__ __device__ complex(const T& re, const T& im);
  __host__ __device__ inline complex<T>& operator*=(const complex<T> z);
  __host__ __device__ inline T real() const volatile { return m_data[0];  }
  __host__ __device__ inline T imag() const volatile { return m_data[1];  }
  __host__ __device__ inline T real() const { return m_data[0];  }
  __host__ __device__ inline T imag() const { return m_data[1];  }
  __host__ __device__ inline void real(T re) volatile { m_data[0] = re;  }
  __host__ __device__ inline void imag(T im) volatile { m_data[1] = im;  }
  __host__ __device__ inline void real(T re) { m_data[0] = re;  }
  __host__ __device__ inline void imag(T im) { m_data[1] = im;  }

 private:
  T m_data[2];
};
template <typename T> inline __host__ __device__ complex<T>::complex(const T& re, const T& im) {
  real(re);
  imag(im);
}
template <typename T>
__host__ __device__ inline complex<T>& complex<T>::operator*=(const complex<T> z) {
  *this = *this * z;
  return *this;
}
template <typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                    lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

template <typename T> __global__ void my_sqrt(T* input, int N) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    input[x] *= input[x];
  }
}
)"};

void test2(){
  using namespace std;
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog,                 // prog
                      template_kernel,       // buffer
                      "template_kernel.cu",  // name
                      0, nullptr, nullptr);
  UPTKDeviceProp props;
  int device = 0;
  CUDACHECK(UPTKGetDeviceProperties(&props, device));
  string sarg = string("--gpu-architecture=compute_")
    + to_string(props.major) + to_string(props.minor);
  const char* options[] = {sarg.c_str()};

  std::vector<std::string> name_expressions;
  name_expressions.push_back("my_sqrt<int>");
  name_expressions.push_back("my_sqrt<float>");
  name_expressions.push_back("my_sqrt<complex<double>>");
  name_expressions.push_back("my_sqrt<complex<double> >");
  for (size_t i = 0; i < name_expressions.size(); i++) {
     EXPECT_EQ(nvrtcAddNameExpression(prog, name_expressions[i].c_str()), NVRTC_SUCCESS);
  }

  nvrtcResult compileResult{nvrtcCompileProgram(prog, 1, options)};

  size_t logSize;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }

  std::map<std::string, std::string> mangled_names;

  for (size_t i = 0; i < name_expressions.size(); i++) {
    const char* mangled_instantiation_cstr;
    EXPECT_EQ(nvrtcGetLoweredName(prog, name_expressions[i].c_str(), &mangled_instantiation_cstr), NVRTC_SUCCESS);
    std::string mangled_name_str = mangled_instantiation_cstr;
    mangled_names[name_expressions[i]] = mangled_name_str;
    EXPECT_GT(mangled_name_str.size(), 0);

  }

  // Match the last two names
  EXPECT_EQ(mangled_names["my_sqrt<complex<double>>"], mangled_names["my_sqrt<complex<double> >"]);
  nvrtcDestroyProgram(&prog);
  EXPECT_EQ(compileResult, NVRTC_SUCCESS);
}

void test3(){
  using namespace std;
  nvrtcProgram prog;
  nvrtcCreateProgram(&prog,                 // prog
                      template_kernel,       // buffer
                      "template_kernel.cu",  // name
                      0, nullptr, nullptr);

  std::string name_expression = "my_sqrt<complex<double> >";

  EXPECT_EQ(nvrtcAddNameExpression(prog, name_expression.c_str()), NVRTC_SUCCESS);

  nvrtcResult compileResult{nvrtcCompileProgram(prog, 0, 0)};

  size_t logSize;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }

  const char* mangled_instantiation_cstr;
  // Verifies if hiprtcGetLoweredName successfully gets the lowered name for named expressions with space
  EXPECT_EQ(nvrtcGetLoweredName(prog, name_expression.c_str(), &mangled_instantiation_cstr), NVRTC_SUCCESS);
  std::string mangled_name_str = mangled_instantiation_cstr;
  // Checks if the fetched lowered name is not empty
  EXPECT_GT(mangled_name_str.size(), 0);

  nvrtcDestroyProgram(&prog);
  EXPECT_EQ(compileResult, NVRTC_SUCCESS);
}

TEST(cudanvrtc, customOptionsTest) {
    test1();
    test2();
    test3();
}
