/******************************************************************************
MPP Final
Dot Product Optimization - Kernel
Jake Webb
******************************************************************************/

#define BLOCK_SIZE 1024
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <math.h>

#include <curand_kernel.h>

#include <stdlib.h>

#include "support.h"

__global__ void monte_carlo_simple(float* x_array, float* y_array, int N, int* count_array) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    float x=x_array[index];
    float y=y_array[index];

    if (index < N) {
        if ((x*x + y*y) < 1) {
            count_array[index] = 1;
        }
    }
}

__global__ void monte_carlo_advanced(float* x_array, float* y_array, int N, int* count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    float x=x_array[index];
    float y=y_array[index];

    if (index < N) {
        if ((x*x + y*y) < 1) {
            atomicAdd(count, 1);
        }
    }
}