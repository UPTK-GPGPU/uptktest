// ===--- c2c_1d_inplace.cu ----------------------------------*- CUDA -*---===//
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

void c2c_1d_inplace()
{
  UPTKfftHandle plan_fwd;
  float2 forward_idata_h[14];
  set_value((float *)forward_idata_h, 14);
  set_value((float *)forward_idata_h + 14, 14);

  float2 *data_d;
  UPTKMalloc(&data_d, 2 * sizeof(float2) * 7);
  UPTKMemcpy(data_d, forward_idata_h, 2 * sizeof(float2) * 7, UPTKMemcpyHostToDevice);

  UPTKfftPlan1d(&plan_fwd, 7, UPTKFFT_C2C, 2);
  UPTKfftExecC2C(plan_fwd, data_d, data_d, UPTKFFT_FORWARD);
  UPTKDeviceSynchronize();
  float2 forward_odata_h[14];
  UPTKMemcpy(forward_odata_h, data_d, 2 * sizeof(float2) * 7, UPTKMemcpyDeviceToHost);

  float2 forward_odata_ref[14];
  forward_odata_ref[0] = float2{42, 49};
  forward_odata_ref[1] = float2{-21.5356, 7.53565};
  forward_odata_ref[2] = float2{-12.5823, -1.41769};
  forward_odata_ref[3] = float2{-8.5977, -5.4023};
  forward_odata_ref[4] = float2{-5.4023, -8.5977};
  forward_odata_ref[5] = float2{-1.41769, -12.5823};
  forward_odata_ref[6] = float2{7.53565, -21.5356};
  forward_odata_ref[7] = float2{42, 49};
  forward_odata_ref[8] = float2{-21.5356, 7.53565};
  forward_odata_ref[9] = float2{-12.5823, -1.41769};
  forward_odata_ref[10] = float2{-8.5977, -5.4023};
  forward_odata_ref[11] = float2{-5.4023, -8.5977};
  forward_odata_ref[12] = float2{-1.41769, -12.5823};
  forward_odata_ref[13] = float2{7.53565, -21.5356};

  UPTKfftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 14);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 14);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 14)

  UPTKfftHandle plan_bwd;
  UPTKfftPlan1d(&plan_bwd, 7, UPTKFFT_C2C, 2);
  UPTKfftExecC2C(plan_bwd, data_d, data_d, UPTKFFT_INVERSE);
  UPTKDeviceSynchronize();
  float2 backward_odata_h[14];
  UPTKMemcpy(backward_odata_h, data_d, 2 * sizeof(float2) * 7, UPTKMemcpyDeviceToHost);

  float2 backward_odata_ref[14];
  backward_odata_ref[0] = float2{0, 7};
  backward_odata_ref[1] = float2{14, 21};
  backward_odata_ref[2] = float2{28, 35};
  backward_odata_ref[3] = float2{42, 49};
  backward_odata_ref[4] = float2{56, 63};
  backward_odata_ref[5] = float2{70, 77};
  backward_odata_ref[6] = float2{84, 91};
  backward_odata_ref[7] = float2{0, 7};
  backward_odata_ref[8] = float2{14, 21};
  backward_odata_ref[9] = float2{28, 35};
  backward_odata_ref[10] = float2{42, 49};
  backward_odata_ref[11] = float2{56, 63};
  backward_odata_ref[12] = float2{70, 77};
  backward_odata_ref[13] = float2{84, 91};

  UPTKFree(data_d);
  UPTKfftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 14);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 14);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 14);
}

TEST(cufft_runable, c2c_1d_inplace)
{
#define FUNC c2c_1d_inplace
  FUNC();
  UPTKDeviceSynchronize();
}
