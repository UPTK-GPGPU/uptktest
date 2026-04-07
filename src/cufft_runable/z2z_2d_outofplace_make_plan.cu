// ===--- z2z_2d_outofplace_make_plan.cu ---------------------*- CUDA -*---===//
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

void z2z_2d_outofplace_make_plan()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  double2 forward_idata_h[2][5];
  set_value((double *)forward_idata_h, 20);

  double2 *forward_idata_d;
  double2 *forward_odata_d;
  double2 *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(double2) * 10);
  UPTKMalloc(&forward_odata_d, sizeof(double2) * 10);
  UPTKMalloc(&backward_odata_d, sizeof(double2) * 10);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(double2) * 10, UPTKMemcpyHostToDevice);

  size_t workSize;
  UPTKfftMakePlan2d(plan_fwd, 2, 5, UPTKFFT_Z2Z, &workSize);
  UPTKfftExecZ2Z(plan_fwd, forward_idata_d, forward_odata_d, UPTKFFT_FORWARD);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[10];
  UPTKMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 10, UPTKMemcpyDeviceToHost);

  double2 forward_odata_ref[10];
  forward_odata_ref[0] = double2{90, 100};
  forward_odata_ref[1] = double2{-23.7638, 3.76382};
  forward_odata_ref[2] = double2{-13.2492, -6.7508};
  forward_odata_ref[3] = double2{-6.7508, -13.2492};
  forward_odata_ref[4] = double2{3.76382, -23.7638};
  forward_odata_ref[5] = double2{-50, -50};
  forward_odata_ref[6] = double2{0, 0};
  forward_odata_ref[7] = double2{0, 0};
  forward_odata_ref[8] = double2{0, 0};
  forward_odata_ref[9] = double2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 10);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 10);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 10);

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlan2d(plan_bwd, 2, 5, UPTKFFT_Z2Z, &workSize);
  UPTKfftExecZ2Z(plan_bwd, forward_odata_d, backward_odata_d, UPTKFFT_INVERSE);
  UPTKDeviceSynchronize();
  double2 backward_odata_h[10];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(double2) * 10, UPTKMemcpyDeviceToHost);

  double2 backward_odata_ref[10];
  backward_odata_ref[0] = double2{0, 10};
  backward_odata_ref[1] = double2{20, 30};
  backward_odata_ref[2] = double2{40, 50};
  backward_odata_ref[3] = double2{60, 70};
  backward_odata_ref[4] = double2{80, 90};
  backward_odata_ref[5] = double2{100, 110};
  backward_odata_ref[6] = double2{120, 130};
  backward_odata_ref[7] = double2{140, 150};
  backward_odata_ref[8] = double2{160, 170};
  backward_odata_ref[9] = double2{180, 190};

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 10);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 10);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 10);
}

TEST(cufft_runable, z2z_2d_outofplace_make_plan)
{
#define FUNC z2z_2d_outofplace_make_plan
  FUNC();
  UPTKDeviceSynchronize();
}
