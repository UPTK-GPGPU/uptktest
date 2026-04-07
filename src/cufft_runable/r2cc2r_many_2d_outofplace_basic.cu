// ===--- r2cc2r_many_2d_outofplace_basic.cu -----------------*- CUDA -*---===//
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

void r2cc2r_many_2d_outofplace_basic()
{
  UPTKfftHandle plan_fwd;
  float forward_idata_h[2 /*n0*/ * 3 /*n1*/ * 2 /*batch*/];
  set_value(forward_idata_h, 6);
  set_value(forward_idata_h + 6, 6);

  float *forward_idata_d;
  float2 *forward_odata_d;
  float *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(float) * 2 * 3 * 2);
  UPTKMalloc(&forward_odata_d, 2 * 2 * sizeof(float2) * (3 / 2 + 1));
  UPTKMalloc(&backward_odata_d, sizeof(float) * 2 * 3 * 2);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(float) * 2 * 3 * 2, UPTKMemcpyHostToDevice);

  int n[2] = {2, 3};
  UPTKfftPlanMany(&plan_fwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_R2C, 2);
  UPTKfftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[8];
  UPTKMemcpy(forward_odata_h, forward_odata_d, 2 * 2 * sizeof(float2) * (3 / 2 + 1), UPTKMemcpyDeviceToHost);

  float2 forward_odata_ref[8];
  forward_odata_ref[0] = float2{15, 0};
  forward_odata_ref[1] = float2{-3, 1.73205};
  forward_odata_ref[2] = float2{-9, 0};
  forward_odata_ref[3] = float2{0, 0};
  forward_odata_ref[4] = float2{15, 0};
  forward_odata_ref[5] = float2{-3, 1.73205};
  forward_odata_ref[6] = float2{-9, 0};
  forward_odata_ref[7] = float2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 8);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 8);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 8);

  UPTKfftHandle plan_bwd;
  UPTKfftPlanMany(&plan_bwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_C2R, 2);
  UPTKfftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  UPTKDeviceSynchronize();
  float backward_odata_h[12];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 12, UPTKMemcpyDeviceToHost);

  float backward_odata_ref[12];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 6;
  backward_odata_ref[2] = 12;
  backward_odata_ref[3] = 18;
  backward_odata_ref[4] = 24;
  backward_odata_ref[5] = 30;
  backward_odata_ref[6] = 0;
  backward_odata_ref[7] = 6;
  backward_odata_ref[8] = 12;
  backward_odata_ref[9] = 18;
  backward_odata_ref[10] = 24;
  backward_odata_ref[11] = 30;

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 12);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 12);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 12);
}

TEST(cufft_runable, r2cc2r_many_2d_outofplace_basic)
{
#define FUNC r2cc2r_many_2d_outofplace_basic
  FUNC();
  UPTKDeviceSynchronize();
}
