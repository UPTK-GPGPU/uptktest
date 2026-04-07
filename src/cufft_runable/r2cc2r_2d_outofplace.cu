// ===--- r2cc2r_2d_outofplace.cu ----------------------------*- CUDA -*---===//
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

void r2cc2r_2d_outofplace()
{
  UPTKfftHandle plan_fwd;
  float forward_idata_h[4][5];
  set_value((float *)forward_idata_h, 20);

  float *forward_idata_d;
  float2 *forward_odata_d;
  float *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(float) * 20);
  UPTKMalloc(&forward_odata_d, sizeof(float2) * (5 / 2 + 1) * 4);
  UPTKMalloc(&backward_odata_d, sizeof(float) * 20);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(float) * 20, UPTKMemcpyHostToDevice);

  UPTKfftPlan2d(&plan_fwd, 4, 5, UPTKFFT_R2C);
  UPTKfftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[12];
  UPTKMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * (5 / 2 + 1) * 4, UPTKMemcpyDeviceToHost);

  float2 forward_odata_ref[12];
  forward_odata_ref[0] = float2{190, 0};
  forward_odata_ref[1] = float2{-10, 13.7638};
  forward_odata_ref[2] = float2{-10, 3.2492};
  forward_odata_ref[3] = float2{-50, 50};
  forward_odata_ref[4] = float2{0, 0};
  forward_odata_ref[5] = float2{0, 0};
  forward_odata_ref[6] = float2{-50, 0};
  forward_odata_ref[7] = float2{0, 0};
  forward_odata_ref[8] = float2{0, 0};
  forward_odata_ref[9] = float2{-50, -50};
  forward_odata_ref[10] = float2{0, 0};
  forward_odata_ref[11] = float2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 12);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 12);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 12);

  UPTKfftHandle plan_bwd;
  UPTKfftPlan2d(&plan_bwd, 4, 5, UPTKFFT_C2R);
  UPTKfftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  UPTKDeviceSynchronize();
  float backward_odata_h[20];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 20, UPTKMemcpyDeviceToHost);

  float backward_odata_ref[20];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 20;
  backward_odata_ref[2] = 40;
  backward_odata_ref[3] = 60;
  backward_odata_ref[4] = 80;
  backward_odata_ref[5] = 100;
  backward_odata_ref[6] = 120;
  backward_odata_ref[7] = 140;
  backward_odata_ref[8] = 160;
  backward_odata_ref[9] = 180;
  backward_odata_ref[10] = 200;
  backward_odata_ref[11] = 220;
  backward_odata_ref[12] = 240;
  backward_odata_ref[13] = 260;
  backward_odata_ref[14] = 280;
  backward_odata_ref[15] = 300;
  backward_odata_ref[16] = 320;
  backward_odata_ref[17] = 340;
  backward_odata_ref[18] = 360;
  backward_odata_ref[19] = 380;

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 20);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 20);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 20);
}

TEST(cufft_runable, r2cc2r_2d_outofplace)
{
#define FUNC r2cc2r_2d_outofplace
  FUNC();
  UPTKDeviceSynchronize();
}
