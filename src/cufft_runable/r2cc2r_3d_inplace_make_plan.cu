// ===--- r2cc2r_3d_inplace_make_plan.cu ---------------------*- CUDA -*---===//
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

void r2cc2r_3d_inplace_make_plan()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  float forward_idata_h[36];
  set_value(forward_idata_h, 2, 3, 5, 6);

  float *data_d;
  UPTKMalloc(&data_d, sizeof(float) * 36);
  UPTKMemcpy(data_d, forward_idata_h, sizeof(float) * 36, UPTKMemcpyHostToDevice);

  size_t workSize;
  UPTKfftMakePlan3d(plan_fwd, 2, 3, 5, UPTKFFT_R2C, &workSize);
  UPTKfftExecR2C(plan_fwd, data_d, (float2 *)data_d);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[18];
  UPTKMemcpy(forward_odata_h, data_d, sizeof(float) * 36, UPTKMemcpyDeviceToHost);

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
  // print_values(forward_odata_ref, 18)

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlan3d(plan_bwd, 2, 3, 5, UPTKFFT_C2R, &workSize);
  UPTKfftExecC2R(plan_bwd, (float2 *)data_d, data_d);
  UPTKDeviceSynchronize();
  float backward_odata_h[36];
  UPTKMemcpy(backward_odata_h, data_d, sizeof(float) * 36, UPTKMemcpyDeviceToHost);

  float backward_odata_ref[36];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 30;
  backward_odata_ref[2] = 60;
  backward_odata_ref[3] = 90;
  backward_odata_ref[4] = 120;
  backward_odata_ref[5] = 4.8738;
  backward_odata_ref[6] = 150;
  backward_odata_ref[7] = 180;
  backward_odata_ref[8] = 210;
  backward_odata_ref[9] = 240;
  backward_odata_ref[10] = 270;
  backward_odata_ref[11] = 4.8738;
  backward_odata_ref[12] = 300;
  backward_odata_ref[13] = 330;
  backward_odata_ref[14] = 360;
  backward_odata_ref[15] = 390;
  backward_odata_ref[16] = 420;
  backward_odata_ref[17] = 4.8738;
  backward_odata_ref[18] = 450;
  backward_odata_ref[19] = 480;
  backward_odata_ref[20] = 510;
  backward_odata_ref[21] = 540;
  backward_odata_ref[22] = 570;
  backward_odata_ref[23] = 4.8738;
  backward_odata_ref[24] = 600;
  backward_odata_ref[25] = 630;
  backward_odata_ref[26] = 660;
  backward_odata_ref[27] = 690;
  backward_odata_ref[28] = 720;
  backward_odata_ref[29] = 4.8738;
  backward_odata_ref[30] = 750;
  backward_odata_ref[31] = 780;
  backward_odata_ref[32] = 810;
  backward_odata_ref[33] = 840;
  backward_odata_ref[34] = 870;
  backward_odata_ref[35] = 4.8738;

  UPTKFree(data_d);
  UPTKfftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4,
                              6, 7, 8, 9, 10,
                              12, 13, 14, 15, 16,
                              18, 19, 20, 21, 22,
                              24, 25, 26, 27, 28,
                              30, 31, 32, 33, 34};
  compare(backward_odata_ref, backward_odata_h, indices);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, indices);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, indices);
}

TEST(cufft_runable, r2cc2r_3d_inplace_make_plan)
{
#define FUNC r2cc2r_3d_inplace_make_plan
  FUNC();
  UPTKDeviceSynchronize();
}
