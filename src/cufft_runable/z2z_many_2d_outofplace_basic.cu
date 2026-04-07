// ===--- z2z_many_2d_outofplace_basic.cu --------------------*- CUDA -*---===//
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

void z2z_many_2d_outofplace_basic()
{
  UPTKfftHandle plan_fwd;
  double2 forward_idata_h[2 /*n0*/ * 3 /*n1*/ * 2 /*batch*/];
  set_value((double *)forward_idata_h, 12);
  set_value((double *)forward_idata_h + 12, 12);

  double2 *forward_idata_d;
  double2 *forward_odata_d;
  double2 *backward_odata_d;
  UPTKMalloc(&forward_idata_d, sizeof(double2) * 12);
  UPTKMalloc(&forward_odata_d, sizeof(double2) * 12);
  UPTKMalloc(&backward_odata_d, sizeof(double2) * 12);
  UPTKMemcpy(forward_idata_d, forward_idata_h, sizeof(double2) * 12, UPTKMemcpyHostToDevice);

  int n[2] = {2, 3};
  UPTKfftPlanMany(&plan_fwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_Z2Z, 2);
  UPTKfftExecZ2Z(plan_fwd, forward_idata_d, forward_odata_d, UPTKFFT_FORWARD);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[12];
  UPTKMemcpy(forward_odata_h, forward_odata_d, sizeof(double2) * 12, UPTKMemcpyDeviceToHost);

  double2 forward_odata_ref[12];
  forward_odata_ref[0] = double2{30, 36};
  forward_odata_ref[1] = double2{-9.4641, -2.5359};
  forward_odata_ref[2] = double2{-2.5359, -9.4641};
  forward_odata_ref[3] = double2{-18, -18};
  forward_odata_ref[4] = double2{0, 0};
  forward_odata_ref[5] = double2{0, 0};
  forward_odata_ref[6] = double2{30, 36};
  forward_odata_ref[7] = double2{-9.4641, -2.5359};
  forward_odata_ref[8] = double2{-2.5359, -9.4641};
  forward_odata_ref[9] = double2{-18, -18};
  forward_odata_ref[10] = double2{0, 0};
  forward_odata_ref[11] = double2{0, 0};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 12);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 12);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 12);

  UPTKfftHandle plan_bwd;
  UPTKfftPlanMany(&plan_bwd, 2, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_Z2Z, 2);
  UPTKfftExecZ2Z(plan_bwd, forward_odata_d, backward_odata_d, UPTKFFT_INVERSE);
  UPTKDeviceSynchronize();
  double2 backward_odata_h[12];
  UPTKMemcpy(backward_odata_h, backward_odata_d, sizeof(double2) * 12, UPTKMemcpyDeviceToHost);

  double2 backward_odata_ref[12];
  backward_odata_ref[0] = double2{0, 6};
  backward_odata_ref[1] = double2{12, 18};
  backward_odata_ref[2] = double2{24, 30};
  backward_odata_ref[3] = double2{36, 42};
  backward_odata_ref[4] = double2{48, 54};
  backward_odata_ref[5] = double2{60, 66};
  backward_odata_ref[6] = double2{0, 6};
  backward_odata_ref[7] = double2{12, 18};
  backward_odata_ref[8] = double2{24, 30};
  backward_odata_ref[9] = double2{36, 42};
  backward_odata_ref[10] = double2{48, 54};
  backward_odata_ref[11] = double2{60, 66};

  UPTKFree(forward_idata_d);
  UPTKFree(forward_odata_d);
  UPTKFree(backward_odata_d);

  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 12);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 12);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 12);
}

TEST(cufft_runable, z2z_many_2d_outofplace_basic)
{
#define FUNC z2z_many_2d_outofplace_basic
  FUNC();
  UPTKDeviceSynchronize();
}
