// ===--- d2zz2d_3d_inplace.cu -------------------------------*- CUDA -*---===//
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

void d2zz2d_3d_inplace()
{
  UPTKfftHandle plan_fwd;
  double forward_idata_h[36];
  set_value(forward_idata_h, 2, 3, 5, 6);

  double *data_d;
  UPTKMalloc(&data_d, sizeof(double) * 36);
  UPTKMemcpy(data_d, forward_idata_h, sizeof(double) * 36, UPTKMemcpyHostToDevice);

  UPTKfftPlan3d(&plan_fwd, 2, 3, 5, UPTKFFT_D2Z);
  UPTKfftExecD2Z(plan_fwd, data_d, (double2 *)data_d);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[18];
  UPTKMemcpy(forward_odata_h, data_d, sizeof(double) * 36, UPTKMemcpyDeviceToHost);

  double2 forward_odata_ref[18];
  forward_odata_ref[0] = double2{435, 0};
  forward_odata_ref[1] = double2{-15, 20.6457};
  forward_odata_ref[2] = double2{-15, 4.8738};
  forward_odata_ref[3] = double2{-75, 43.3013};
  forward_odata_ref[4] = double2{0, 0};
  forward_odata_ref[5] = double2{0, 0};
  forward_odata_ref[6] = double2{-75, -43.3013};
  forward_odata_ref[7] = double2{0, 0};
  forward_odata_ref[8] = double2{0, 0};
  forward_odata_ref[9] = double2{-225, 0};
  forward_odata_ref[10] = double2{0, 0};
  forward_odata_ref[11] = double2{0, 0};
  forward_odata_ref[12] = double2{0, 0};
  forward_odata_ref[13] = double2{0, 0};
  forward_odata_ref[14] = double2{0, 0};
  forward_odata_ref[15] = double2{0, 0};
  forward_odata_ref[16] = double2{0, 0};
  forward_odata_ref[17] = double2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 18);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 18);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 18)

  UPTKfftHandle plan_bwd;
  UPTKfftPlan3d(&plan_bwd, 2, 3, 5, UPTKFFT_Z2D);
  UPTKfftExecZ2D(plan_bwd, (double2 *)data_d, data_d);
  UPTKDeviceSynchronize();
  double backward_odata_h[36];
  UPTKMemcpy(backward_odata_h, data_d, sizeof(double) * 36, UPTKMemcpyDeviceToHost);

  double backward_odata_ref[36];
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

TEST(cufft_runable, d2zz2d_3d_inplace)
{
#define FUNC d2zz2d_3d_inplace
  FUNC();
  UPTKDeviceSynchronize();
}
