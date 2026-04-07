// ===--- d2zz2d_many_1d_outofplace_basic.cu -----------------*- CUDA -*---===//
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

void d2zz2d_many_1d_outofplace_basic()
{
  UPTKfftHandle plan_fwd;
  double forward_idata_h[20];
  set_value(forward_idata_h, 10);
  set_value(forward_idata_h + 10, 10);

  double *forward_idata_d;
  double2 *forward_odata_d;
  double *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(double) * 20);
  UPTKMalloc(&forward_odata_d, 2 * sizeof(double2) * (10 / 2 + 1));
  UPTKMalloc(&backward_odata_d, sizeof(double) * 20);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(double) * 20, UPTKMemcpyHostToDevice);

  int n[1] = {10};
  UPTKfftPlanMany(&plan_fwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_D2Z, 2);
  UPTKfftExecD2Z(plan_fwd, forward_idata_d, forward_odata_d);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[12];
  UPTKMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(double2) * (10 / 2 + 1), UPTKMemcpyDeviceToHost);

  double2 forward_odata_ref[12];
  forward_odata_ref[0] = double2{45, 0};
  forward_odata_ref[1] = double2{-5, 15.3884};
  forward_odata_ref[2] = double2{-5, 6.88191};
  forward_odata_ref[3] = double2{-5, 3.63271};
  forward_odata_ref[4] = double2{-5, 1.6246};
  forward_odata_ref[5] = double2{-5, 0};
  forward_odata_ref[6] = double2{45, 0};
  forward_odata_ref[7] = double2{-5, 15.3884};
  forward_odata_ref[8] = double2{-5, 6.88191};
  forward_odata_ref[9] = double2{-5, 3.63271};
  forward_odata_ref[10] = double2{-5, 1.6246};
  forward_odata_ref[11] = double2{-5, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 12);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 12);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 12);

  UPTKfftHandle plan_bwd;
  UPTKfftPlanMany(&plan_bwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_Z2D, 2);
  UPTKfftExecZ2D(plan_bwd, forward_odata_d, backward_odata_d);
  UPTKDeviceSynchronize();
  double backward_odata_h[20];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(double) * 20, UPTKMemcpyDeviceToHost);

  double backward_odata_ref[20];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 10;
  backward_odata_ref[2] = 20;
  backward_odata_ref[3] = 30;
  backward_odata_ref[4] = 40;
  backward_odata_ref[5] = 50;
  backward_odata_ref[6] = 60;
  backward_odata_ref[7] = 70;
  backward_odata_ref[8] = 80;
  backward_odata_ref[9] = 90;
  backward_odata_ref[10] = 0;
  backward_odata_ref[11] = 10;
  backward_odata_ref[12] = 20;
  backward_odata_ref[13] = 30;
  backward_odata_ref[14] = 40;
  backward_odata_ref[15] = 50;
  backward_odata_ref[16] = 60;
  backward_odata_ref[17] = 70;
  backward_odata_ref[18] = 80;
  backward_odata_ref[19] = 90;

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

TEST(cufft_runable, d2zz2d_many_1d_outofplace_basic)
{
#define FUNC d2zz2d_many_1d_outofplace_basic
  FUNC();
  UPTKDeviceSynchronize();
}
