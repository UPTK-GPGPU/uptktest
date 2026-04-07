// ===--- d2zz2d_2d_outofplace_make_plan.cu ------------------*- CUDA -*---===//
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

void d2zz2d_2d_outofplace_make_plan()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  double forward_idata_h[4][5];
  set_value((double *)forward_idata_h, 20);

  double *forward_idata_d;
  double2 *forward_odata_d;
  double *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(double) * 20);
  UPTKMalloc(&forward_odata_d, sizeof(double2) * (5 / 2 + 1) * 4);
  UPTKMalloc(&backward_odata_d, sizeof(double) * 20);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(double) * 20, UPTKMemcpyHostToDevice);

  size_t workSize;
  UPTKfftMakePlan2d(plan_fwd, 4, 5, UPTKFFT_D2Z, &workSize);
  UPTKfftExecD2Z(plan_fwd, forward_idata_d, forward_odata_d);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[12];
  UPTKMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * (5 / 2 + 1) * 4, UPTKMemcpyDeviceToHost);

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
  // print_values(forward_odata_ref, 12);

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlan2d(plan_bwd, 4, 5, UPTKFFT_Z2D, &workSize);
  UPTKfftExecZ2D(plan_bwd, forward_odata_d, backward_odata_d);
  UPTKDeviceSynchronize();
  double backward_odata_h[20];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 20, UPTKMemcpyDeviceToHost);

  double backward_odata_ref[20];
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

TEST(cufft_runable, d2zz2d_2d_outofplace_make_plan)
{
#define FUNC d2zz2d_2d_outofplace_make_plan
  FUNC();
  UPTKDeviceSynchronize();
}
