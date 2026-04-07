// ===--- d2zz2d_1d_outofplace.cu ----------------------------*- CUDA -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===---------------------------------------------------------------------===//

#include "cufft.h"
#include "common.h"
#include <cstring>
#include <iostream>

void d2zz2d_1d_outofplace()
{
  cufftHandle plan_fwd;
  double forward_idata_h[14];
  set_value(forward_idata_h, 7);
  set_value(forward_idata_h + 7, 7);

  double *forward_idata_d;
  double2 *forward_odata_d;
  double *backward_odata_d;
  cudaMalloc(&forward_idata_d, 2 * sizeof(double) * 7);
  cudaMalloc(&forward_odata_d, 2 * sizeof(double2) * (7 / 2 + 1));
  cudaMalloc(&backward_odata_d, 2 * sizeof(double) * 7);
  cudaMemcpy(forward_idata_d, forward_idata_h, 2 * sizeof(double) * 7, cudaMemcpyHostToDevice);

  cufftPlan1d(&plan_fwd, 7, CUFFT_D2Z, 2);
  cufftExecD2Z(plan_fwd, forward_idata_d, forward_odata_d);
  cudaDeviceSynchronize();
  double2 forward_odata_h[8];
  cudaMemcpy(forward_odata_h, forward_odata_d, 2 * sizeof(double2) * (7 / 2 + 1), cudaMemcpyDeviceToHost);

  double2 forward_odata_ref[8];
  forward_odata_ref[0] = double2{21, 0};
  forward_odata_ref[1] = double2{-3.5, 7.26783};
  forward_odata_ref[2] = double2{-3.5, 2.79116};
  forward_odata_ref[3] = double2{-3.5, 0.798852};
  forward_odata_ref[4] = double2{21, 0};
  forward_odata_ref[5] = double2{-3.5, 7.26783};
  forward_odata_ref[6] = double2{-3.5, 2.79116};
  forward_odata_ref[7] = double2{-3.5, 0.798852};

  cufftDestroy(plan_fwd);

  compare(forward_odata_ref, forward_odata_h, 8);
  // std::cout << "forward_odata_h:" << std::endl;
  // print_values(forward_odata_h, 8);
  // std::cout << "forward_odata_ref:" << std::endl;
  // print_values(forward_odata_ref, 8);

  cufftHandle plan_bwd;
  cufftPlan1d(&plan_bwd, 7, CUFFT_Z2D, 2);
  cufftExecZ2D(plan_bwd, forward_odata_d, backward_odata_d);
  cudaDeviceSynchronize();
  double backward_odata_h[14];
  cudaMemcpy(backward_odata_h, backward_odata_d, 2 * sizeof(double) * 7, cudaMemcpyDeviceToHost);

  double backward_odata_ref[14];
  backward_odata_ref[0] = 0;
  backward_odata_ref[1] = 7;
  backward_odata_ref[2] = 14;
  backward_odata_ref[3] = 21;
  backward_odata_ref[4] = 28;
  backward_odata_ref[5] = 35;
  backward_odata_ref[6] = 42;
  backward_odata_ref[7] = 0;
  backward_odata_ref[8] = 7;
  backward_odata_ref[9] = 14;
  backward_odata_ref[10] = 21;
  backward_odata_ref[11] = 28;
  backward_odata_ref[12] = 35;
  backward_odata_ref[13] = 42;

  cudaFree(forward_idata_d);
  cudaFree(forward_odata_d);
  cudaFree(backward_odata_d);

  cufftDestroy(plan_bwd);

  compare(backward_odata_ref, backward_odata_h, 14);
  // std::cout << "backward_odata_h:" << std::endl;
  // print_values(backward_odata_h, 14);
  // std::cout << "backward_odata_ref:" << std::endl;
  // print_values(backward_odata_ref, 14);
}

TEST(cufft_runable, d2zz2d_1d_outofplace)
{
#define FUNC d2zz2d_1d_outofplace
  FUNC();
  cudaDeviceSynchronize();
}
