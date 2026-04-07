// ===--- z2z_many_1d_inplace_basic.cu -----------------------*- CUDA -*---===//
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

void z2z_many_1d_inplace_basic()
{
  UPTKfftHandle plan_fwd;
  UPTKfftCreate(&plan_fwd);
  double2 forward_idata_h[10];
  set_value((double *)forward_idata_h, 10);
  set_value((double *)forward_idata_h + 10, 10);

  double2 *data_d;
  UPTKMalloc(&data_d, sizeof(double2) * 10);
  UPTKMemcpy(data_d, forward_idata_h, sizeof(double2) * 10, UPTKMemcpyHostToDevice);

  int n[1] = {5};
  size_t workSize;
  UPTKfftMakePlanMany(plan_fwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_Z2Z, 2, &workSize);
  UPTKfftExecZ2Z(plan_fwd, data_d, data_d, UPTKFFT_FORWARD);
  UPTKDeviceSynchronize();
  double2 forward_odata_h[10];
  UPTKMemcpy(forward_odata_h, data_d, sizeof(double2) * 10, UPTKMemcpyDeviceToHost);

  double2 forward_odata_ref[10];
  forward_odata_ref[0] = double2{20, 25};
  forward_odata_ref[1] = double2{-11.8819, 1.88191};
  forward_odata_ref[2] = double2{-6.6246, -3.3754};
  forward_odata_ref[3] = double2{-3.3754, -6.6246};
  forward_odata_ref[4] = double2{1.88191, -11.8819};
  forward_odata_ref[5] = double2{20, 25};
  forward_odata_ref[6] = double2{-11.8819, 1.88191};
  forward_odata_ref[7] = double2{-6.6246, -3.3754};
  forward_odata_ref[8] = double2{-3.3754, -6.6246};
  forward_odata_ref[9] = double2{1.88191, -11.8819};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 10);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 10);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 10)

  UPTKfftHandle plan_bwd;
  UPTKfftCreate(&plan_bwd);
  UPTKfftMakePlanMany(plan_bwd, 1, n, nullptr, 0, 0, nullptr, 0, 0, UPTKFFT_Z2Z, 2, &workSize);
  UPTKfftExecZ2Z(plan_bwd, data_d, data_d, UPTKFFT_INVERSE);
  UPTKDeviceSynchronize();
  double2 backward_odata_h[10];
  UPTKMemcpy(backward_odata_h, data_d, sizeof(double2) * 10, UPTKMemcpyDeviceToHost);

  double2 backward_odata_ref[10];
  backward_odata_ref[0] = double2{0, 5};
  backward_odata_ref[1] = double2{10, 15};
  backward_odata_ref[2] = double2{20, 25};
  backward_odata_ref[3] = double2{30, 35};
  backward_odata_ref[4] = double2{40, 45};
  backward_odata_ref[5] = double2{0, 5};
  backward_odata_ref[6] = double2{10, 15};
  backward_odata_ref[7] = double2{20, 25};
  backward_odata_ref[8] = double2{30, 35};
  backward_odata_ref[9] = double2{40, 45};

  UPTKFree(data_d);
  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 10);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 10);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 10);
}

TEST(cufft_runable, z2z_many_1d_inplace_basic)
{
#define FUNC z2z_many_1d_inplace_basic
  FUNC();
  UPTKDeviceSynchronize();
}
