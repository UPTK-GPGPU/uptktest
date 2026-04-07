// ===--- c2c_2d_outofplace_make_plan.cu ---------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "UPTK_fft.h"
#include "common.h"
#include <cstring>
#include <iostream>

void c2c_2d_outofplace_make_plan()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  float2 forward_idata_h[2][5];
  set_value((float *)forward_idata_h, 20);

  float2 *forward_idata_d;
  float2 *forward_odata_d;
  float2 *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(float2) * 10);
  UPTKMalloc(&forward_odata_d, sizeof(float2) * 10);
  UPTKMalloc(&backward_odata_d, sizeof(float2) * 10);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(float2) * 10, UPTKMemcpyHostToDevice);

  size_t workSize;
  UPTKfftMakePlan2d(plan_fwd, 2, 5, UPTKFFT_C2C, &workSize);
  UPTKfftExecC2C(plan_fwd, forward_idata_d, forward_odata_d, UPTKFFT_FORWARD);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[10];
  UPTKMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * 10, UPTKMemcpyDeviceToHost);

  float2 forward_odata_ref[10];
  forward_odata_ref[0] = float2{90, 100};
  forward_odata_ref[1] = float2{-23.7638, 3.76382};
  forward_odata_ref[2] = float2{-13.2492, -6.7508};
  forward_odata_ref[3] = float2{-6.7508, -13.2492};
  forward_odata_ref[4] = float2{3.76382, -23.7638};
  forward_odata_ref[5] = float2{-50, -50};
  forward_odata_ref[6] = float2{0, 0};
  forward_odata_ref[7] = float2{0, 0};
  forward_odata_ref[8] = float2{0, 0};
  forward_odata_ref[9] = float2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 10);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 10);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 10);

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlan2d(plan_bwd, 2, 5, UPTKFFT_C2C, &workSize);
  UPTKfftExecC2C(plan_bwd, forward_odata_d, backward_odata_d, UPTKFFT_INVERSE);
  UPTKDeviceSynchronize();
  float2 backward_odata_h[10];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(float2) * 10, UPTKMemcpyDeviceToHost);

  float2 backward_odata_ref[10];
  backward_odata_ref[0] = float2{0, 10};
  backward_odata_ref[1] = float2{20, 30};
  backward_odata_ref[2] = float2{40, 50};
  backward_odata_ref[3] = float2{60, 70};
  backward_odata_ref[4] = float2{80, 90};
  backward_odata_ref[5] = float2{100, 110};
  backward_odata_ref[6] = float2{120, 130};
  backward_odata_ref[7] = float2{140, 150};
  backward_odata_ref[8] = float2{160, 170};
  backward_odata_ref[9] = float2{180, 190};

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 10);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 10);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 10);
}

TEST(cufft_runable, c2c_2d_outofplace_make_plan)
{
#define FUNC c2c_2d_outofplace_make_plan
  FUNC();
  UPTKDeviceSynchronize();
}
