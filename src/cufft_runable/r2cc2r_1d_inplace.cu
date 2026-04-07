// ===--- r2cc2r_1d_inplace.cu -------------------------------*- CUDA -*---===//
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

void r2cc2r_1d_inplace()
{
  UPTKfftHandle plan_fwd;
  float forward_idata_h[16];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 8, 7);

  float *data_d;
  UPTKMalloc(&data_d, sizeof(float) * 16);
  UPTKMemcpy(data_d, forward_idata_h, sizeof(float) * 16, UPTKMemcpyHostToDevice);

  UPTKfftPlan1d(&plan_fwd, 7, UPTKFFT_R2C, 2);
  UPTKfftExecR2C(plan_fwd, data_d, (float2 *)data_d);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[8];
  UPTKMemcpy(forward_odata_h, data_d, sizeof(float) * 16, UPTKMemcpyDeviceToHost);

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
  // print_values(forward_odata_ref, 8)

  UPTKfftHandle plan_bwd;
  UPTKfftPlan1d(&plan_bwd, 7, UPTKFFT_C2R, 2);
  UPTKfftExecC2R(plan_bwd, (float2 *)data_d, data_d);
  UPTKDeviceSynchronize();
  float backward_odata_h[16];
  UPTKMemcpy(backward_odata_h, data_d, sizeof(float) * 16, UPTKMemcpyDeviceToHost);

  float backward_odata_ref[16];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0.798852;
  backward_odata_ref[8] = 0;
  backward_odata_ref[9] = 7;
  backward_odata_ref[10] = 14;
  backward_odata_ref[11] = 21;
  backward_odata_ref[12] = 28;
  backward_odata_ref[13] = 35;
  backward_odata_ref[14] = 42;
  backward_odata_ref[15] = 0.798852;

  UPTKFree(data_d);
  UPTKfftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6,
                              8, 9, 10, 11, 12, 13, 14};
  compare(backward_odata_ref, backward_odata_h, indices);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, indices);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, indices);
}

TEST(cufft_runable, r2cc2r_1d_inplace)
{
#define FUNC r2cc2r_1d_inplace
  FUNC();
  UPTKDeviceSynchronize();
}
