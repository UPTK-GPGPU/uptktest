// ===--- d2zz2d_many_1d_inplace_basic.cu --------------------*- CUDA -*---===//
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

void d2zz2d_many_1d_inplace_basic()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  double forward_idata_h[24];
  set_value(forward_idata_h, 10);
  set_value(forward_idata_h + 12, 10);

  double *data_d;
  UPTKMalloc(&data_d, sizeof(double) * 24);
  UPTKMemcpy(data_d, forward_idata_h, sizeof(double) * 24, UPTKMemcpyHostToDevice);

  int n[1] = {10};
  size_t workSize;
  UPTKfftMakePlanMany(plan_fwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_D2Z, 2, &workSize);
  UPTKfftExecD2Z(plan_fwd, data_d, (double2 *)data_d);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[12];
  UPTKMemcpy(forward_odata_h, data_d, sizeof(double) * 24, UPTKMemcpyDeviceToHost);

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
  // print_values(forward_odata_ref, 12)

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlanMany(plan_bwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_Z2D, 2, &workSize);
  UPTKfftExecZ2D(plan_bwd, (double2 *)data_d, data_d);
  UPTKDeviceSynchronize();
  double backward_odata_h[24];
  UPTKMemcpy(backward_odata_h, data_d, sizeof(double) * 24, UPTKMemcpyDeviceToHost);

  double backward_odata_ref[24];
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
  backward_odata_ref[10] = -5;
  backward_odata_ref[11] = 0;
  backward_odata_ref[12] = 0;
  backward_odata_ref[13] = 10;
  backward_odata_ref[14] = 20;
  backward_odata_ref[15] = 30;
  backward_odata_ref[16] = 40;
  backward_odata_ref[17] = 50;
  backward_odata_ref[18] = 60;
  backward_odata_ref[19] = 70;
  backward_odata_ref[20] = 80;
  backward_odata_ref[21] = 90;
  backward_odata_ref[22] = -5;
  backward_odata_ref[23] = 0;

  UPTKFree(data_d);
  UPTKfftDestroy(plan_bwd);

  std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  compare(backward_odata_ref, backward_odata_h, indices);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, indices);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, indices);
}

TEST(cufft_runable, d2zz2d_many_1d_inplace_basic)
{
#define FUNC d2zz2d_many_1d_inplace_basic
  FUNC();
  UPTKDeviceSynchronize();
}
