// ===--- r2cc2r_1d_outofplace.cu ----------------------------*- CUDA -*---===//
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

void r2cc2r_1d_outofplace()
{
  UPTKfftHandle plan_fwd;
  float forward_idata_h[14];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 7, 7);

  float *forward_idata_d;
  float2 *forward_odata_d;
  float *backward_odata_d;
  UPTKMalloc(&forward_idata_d, 2 * sizeof(float) * 7);
  UPTKMalloc(&forward_odata_d, 2 * sizeof(float2) * (7 / 2 + 1));
  UPTKMalloc(&backward_odata_d, 2 * sizeof(float) * 7);
  UPTKMemcpy(forward_idata_d, forward_idata_h, 2 * sizeof(float) * 7, UPTKMemcpyHostToDevice);

  UPTKfftPlan1d(&plan_fwd, 7, UPTKFFT_R2C, 2);
  UPTKfftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[8];
  UPTKMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(float2) * (7 / 2 + 1), UPTKMemcpyDeviceToHost);

  float2 forward_odata_ref[8];
  forward_odata_ref[0] = float2{21, 0};
  forward_odata_ref[1] = float2{-3.5, 7.26783};
  forward_odata_ref[2] = float2{-3.5, 2.79116};
  forward_odata_ref[3] = float2{-3.5, 0.798852};
  forward_odata_ref[4] = float2{21, 0};
  forward_odata_ref[5] = float2{-3.5, 7.26783};
  forward_odata_ref[6] = float2{-3.5, 2.79116};
  forward_odata_ref[7] = float2{-3.5, 0.798852};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 8);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 8);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 8);

  UPTKfftHandle plan_bwd;
  UPTKfftPlan1d(&plan_bwd, 7, UPTKFFT_C2R, 2);
  UPTKfftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  UPTKDeviceSynchronize();
  float backward_odata_h[14];
  UPTKMemcpy(backward_odata_h, backward_odata_d, 2 * sizeof(float) * 7, UPTKMemcpyDeviceToHost);

  float backward_odata_ref[14];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0;
  backward_odata_ref[8] = 7;
  backward_odata_ref[9] = 14;
  backward_odata_ref[10] = 21;
  backward_odata_ref[11] = 28;
  backward_odata_ref[12] = 35;
  backward_odata_ref[13] = 42;

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 14);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 14);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 14);
}

TEST(cufft_runable, r2cc2r_1d_outofplace)
{
#define FUNC r2cc2r_1d_outofplace
  FUNC();
  UPTKDeviceSynchronize();
}
