// ===--- d2zz2d_2d_inplace_make_plan.cu ---------------------*- CUDA -*---===//
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

void d2zz2d_2d_inplace_make_plan()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  double forward_idata_h[24];
  set_value(forward_idata_h, 4, 5, 6);

  double *data_d;
  UPTKMalloc(&data_d, sizeof(double) * 24);
  UPTKMemcpy(data_d, forward_idata_h, sizeof(double) * 24, UPTKMemcpyHostToDevice);

  size_t workSize;
  UPTKfftMakePlan2d(plan_fwd, 4, 5, UPTKFFT_D2Z, &workSize);
  UPTKfftExecD2Z(plan_fwd, data_d, (double2 *)data_d);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[12];
  UPTKMemcpy(forward_odata_h, data_d, sizeof(double) * 24, UPTKMemcpyDeviceToHost);

  double2 forward_odata_ref[12];
  forward_odata_ref[0] = double2{190, 0};
  forward_odata_ref[1] = double2{-10, 13.7638};
  forward_odata_ref[2] = double2{-10, 3.2492};
  forward_odata_ref[3] = double2{-50, 50};
  forward_odata_ref[4] = double2{0, 0};
  forward_odata_ref[5] = double2{0, 0};
  forward_odata_ref[6] = double2{-50, 0};
  forward_odata_ref[7] = double2{0, 0};
  forward_odata_ref[8] = double2{0, 0};
  forward_odata_ref[9] = double2{-50, -50};
  forward_odata_ref[10] = double2{0, 0};
  forward_odata_ref[11] = double2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 12);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 12);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 12)

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlan2d(plan_bwd, 4, 5, UPTKFFT_Z2D, &workSize);
  UPTKfftExecZ2D(plan_bwd, (double2 *)data_d, data_d);
  UPTKDeviceSynchronize();
  double backward_odata_h[24];
  UPTKMemcpy(backward_odata_h, data_d, sizeof(double) * 24, UPTKMemcpyDeviceToHost);

  double backward_odata_ref[24];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 20;
  backward_odata_ref[2] = 40;
  backward_odata_ref[3] = 60;
  backward_odata_ref[4] = 80;
  backward_odata_ref[5] = 3.2492;
  backward_odata_ref[6] = 100;
  backward_odata_ref[7] = 120;
  backward_odata_ref[8] = 140;
  backward_odata_ref[9] = 160;
  backward_odata_ref[10] = 180;
  backward_odata_ref[11] = 3.2492;
  backward_odata_ref[12] = 200;
  backward_odata_ref[13] = 220;
  backward_odata_ref[14] = 240;
  backward_odata_ref[15] = 260;
  backward_odata_ref[16] = 280;
  backward_odata_ref[17] = 3.2492;
  backward_odata_ref[18] = 300;
  backward_odata_ref[19] = 320;
  backward_odata_ref[20] = 340;
  backward_odata_ref[21] = 360;
  backward_odata_ref[22] = 380;
  backward_odata_ref[23] = 3.2492;

  UPTKFree(data_d);
  UPTKfftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4,
                              6, 7, 8, 9, 10,
                              12, 13, 14, 15, 16,
                              18, 19, 20, 21, 22};
  compare(backward_odata_ref, backward_odata_h, indices);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, indices);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, indices);
}

TEST(cufft_runable, d2zz2d_2d_inplace_make_plan)
{
#define FUNC d2zz2d_2d_inplace_make_plan
  FUNC();
  UPTKDeviceSynchronize();
}
