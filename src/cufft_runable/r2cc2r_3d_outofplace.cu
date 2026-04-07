// ===--- r2cc2r_3d_outofplace.cu ----------------------------*- CUDA -*---===//
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

void r2cc2r_3d_outofplace()
{
  UPTKfftHandle plan_fwd;
  float forward_idata_h[2][3][5];
  set_value((float *)forward_idata_h, 30);

  float *forward_idata_d;
  float2 *forward_odata_d;
  float *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(float) * 30);
  UPTKMalloc(&forward_odata_d, sizeof(float2) * (5 / 2 + 1) * 2 * 3);
  UPTKMalloc(&backward_odata_d, sizeof(float) * 30);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(float) * 30, UPTKMemcpyHostToDevice);

  UPTKfftPlan3d(&plan_fwd, 2, 3, 5, UPTKFFT_R2C);
  UPTKfftExecR2C(plan_fwd, forward_idata_d, forward_odata_d);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[18];
  UPTKMemcpy(forward_odata_h, forward_odata_d, sizeof(float2) * (5 / 2 + 1) * 2 * 3, UPTKMemcpyDeviceToHost);

  float2 forward_odata_ref[18];
  forward_odata_ref[0] = float2{435, 0};
  forward_odata_ref[1] = float2{-15, 20.6457};
  forward_odata_ref[2] = float2{-15, 4.8738};
  forward_odata_ref[3] = float2{-75, 43.3013};
  forward_odata_ref[4] = float2{0, 0};
  forward_odata_ref[5] = float2{0, 0};
  forward_odata_ref[6] = float2{-75, -43.3013};
  forward_odata_ref[7] = float2{0, 0};
  forward_odata_ref[8] = float2{0, 0};
  forward_odata_ref[9] = float2{-225, 0};
  forward_odata_ref[10] = float2{0, 0};
  forward_odata_ref[11] = float2{0, 0};
  forward_odata_ref[12] = float2{0, 0};
  forward_odata_ref[13] = float2{0, 0};
  forward_odata_ref[14] = float2{0, 0};
  forward_odata_ref[15] = float2{0, 0};
  forward_odata_ref[16] = float2{0, 0};
  forward_odata_ref[17] = float2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 18);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 18);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 18);

  UPTKfftHandle plan_bwd;
  UPTKfftPlan3d(&plan_bwd, 2, 3, 5, UPTKFFT_C2R);
  UPTKfftExecC2R(plan_bwd, forward_odata_d, backward_odata_d);
  UPTKDeviceSynchronize();
  float backward_odata_h[30];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(float) * 30, UPTKMemcpyDeviceToHost);

  float backward_odata_ref[30];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 30;
  backward_odata_ref[2] = 60;
  backward_odata_ref[3] = 90;
  backward_odata_ref[4] = 120;
  backward_odata_ref[5] = 150;
  backward_odata_ref[6] = 180;
  backward_odata_ref[7] = 210;
  backward_odata_ref[8] = 240;
  backward_odata_ref[9] = 270;
  backward_odata_ref[10] = 300;
  backward_odata_ref[11] = 330;
  backward_odata_ref[12] = 360;
  backward_odata_ref[13] = 390;
  backward_odata_ref[14] = 420;
  backward_odata_ref[15] = 450;
  backward_odata_ref[16] = 480;
  backward_odata_ref[17] = 510;
  backward_odata_ref[18] = 540;
  backward_odata_ref[19] = 570;
  backward_odata_ref[20] = 600;
  backward_odata_ref[21] = 630;
  backward_odata_ref[22] = 660;
  backward_odata_ref[23] = 690;
  backward_odata_ref[24] = 720;
  backward_odata_ref[25] = 750;
  backward_odata_ref[26] = 780;
  backward_odata_ref[27] = 810;
  backward_odata_ref[28] = 840;
  backward_odata_ref[29] = 870;

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 30);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 30);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 30);
}

TEST(cufft_runable, r2cc2r_3d_outofplace)
{
#define FUNC r2cc2r_3d_outofplace
  FUNC();
  UPTKDeviceSynchronize();
}
