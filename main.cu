/******************************************************************************
MPP Final - Summer 2025
Monte Carlo Optimization - Main
Jake Webb
******************************************************************************/

// Estimate pi using the monte carlo method

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>

#include <stdlib.h>


#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    /* Init Main ------------------------------------------------------------*/
    Timer timer;
    cudaError_t cuda_ret;
    srand(time(0));
    
    unsigned N;
    if(argc == 1) {
        N = 100000000;
    } else if(argc == 2) {
        N = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./monte-carlo        # Input of size 100,000,000 is used"
           "\n    Usage: ./monte-carlo <m>    # Input of size m is used"
           "\n");
        exit(0);
    }
    
    printf("Input Size: %d\n\n", N);
    
    /* Simple Serial Code ---------------------------------------------------*/
    // Method 1
    // Generate an X and Y vector of random numbers
    // Loop through the vectors, calculate
    
    // Init--------------------------------------------------------------------
    printf("--Simple Serial--\n");
    printf("Setting up... ");
    startTime(&timer);
    
    float pi_estimate_ss;
    int num_inside_circle = 0;
    
    float *point_x_array;
    float *point_y_array;
    bool *inside_circle_array;

    point_x_array = (float*) malloc( sizeof(float)*N );
    point_y_array = (float*) malloc( sizeof(float)*N );
    inside_circle_array = (bool*) malloc( sizeof(bool)*N );
    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));
    
    // Compute-----------------------------------------------------------------
    printf("Computing... ");
    startTime(&timer);
    
    // Random point loop
    for(int i = 0 ; i < N; i++) {
        point_x_array[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        point_y_array[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    }
    
    // Is point inside circle
    for(int i = 0; i < N; i++) {
        if (point_x_array[i]*point_x_array[i] + point_y_array[i]*point_y_array[i] < 1) {
            inside_circle_array[i] = true;
        }
    }
    
    // If inside circle, then increase count
    for (int i = 0; i < N; i++) {
        if (inside_circle_array[i] == true) {
            num_inside_circle += 1;
        }
    }
    
    // Estimate pi
    pi_estimate_ss = 4.0 * (float)num_inside_circle / (float)N;    
    
    // Output------------------------------------------------------------------
    stopTime(&timer);  printf(" %f seconds\n", elapsedTime(timer));
    printf("Pi Estimate: %.6f\n\n", pi_estimate_ss);
    
    // Free--------------------------------------------------------------------
    free(point_x_array);
    free(point_y_array);
    free(inside_circle_array);
    
    /* Advanced Serial Code -------------------------------------------------*/
    // Method 2
    // Better memory efficiency, uses single points instead of vectors
    
    // Init--------------------------------------------------------------------
    printf("--Advanced Serial--\n");
    printf("Setting up... ");
    startTime(&timer);
    
    float pi_estimate_as; 
    num_inside_circle = 0; // same memory as before
    
    float point_x;
    float point_y;
    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));
    
    // Compute-----------------------------------------------------------------
    printf("Computing... ");
    startTime(&timer);
    
    // Main Loop
    for(int i = 0; i < N; i++) {
        point_x = ((float)rand() / (float)RAND_MAX) * 2 - 1; // Calculate from -1 to 1
        point_y = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        
        if (point_x*point_x + point_y*point_y < 1) {
            num_inside_circle += 1;
        }
    }

    // Estimate Pi
    pi_estimate_as = 4.0 * (float)num_inside_circle / (float)N;
    
    // Output------------------------------------------------------------------
    stopTime(&timer);   printf(" %f s\n", elapsedTime(timer));
    printf("Pi Estimate: %.6f\n\n", pi_estimate_as);
    
    // Free--------------------------------------------------------------------
    // Nothing to free
    

    /* CUDA Implementation---------------------------------------------------*/
    // Method 3
    // Threading
    
    // Init--------------------------------------------------------------------
    printf("--Simple CUDA--\n");
    printf("Setting up...                       ");
    startTime(&timer);
    
    float pi_estimate_sc;
    num_inside_circle = 0;
    
    float *x_array_h, *y_array_h;
    float *x_array_d, *y_array_d;
    int *count_array_h, *count_array_d;
    
    x_array_h = (float*) malloc( sizeof(float)*N );
    y_array_h = (float*) malloc( sizeof(float)*N ); 
    count_array_h = (int*) malloc( sizeof(int)*N ); 
    
    for(int i = 0 ; i < N; i++) {
        x_array_h[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        y_array_h[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        count_array_h[i] = 0;
    }      
    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));
    
    // CUDA--------------------------------------------------------------------
    
    // Allocate device memory
    printf("Allocating device memory...         ");    
    startTime(&timer);
    
    cuda_ret = cudaMalloc((void**) &x_array_d, N*sizeof(float));
    if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory for x array ");
    
    cuda_ret = cudaMalloc((void**) &y_array_d, N*sizeof(float));
    if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory for y array ");     
 
    cuda_ret = cudaMalloc((void**) &count_array_d, N*sizeof(int));
    if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory for count array ");     
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));
    
    // Copy data from host to device
    printf("Copying data from host to device... ");    
    startTime(&timer);

    cuda_ret = cudaMemcpy(x_array_d, x_array_h, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) printf("Unable to copy x array to device ");    

    cuda_ret = cudaMemcpy(y_array_d, y_array_h, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) printf("Unable to copy y array to device ");

    cuda_ret = cudaMemcpy(count_array_d, count_array_h, N*sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) printf("Unable to copy count array to device ");

    cudaDeviceSynchronize();    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));    
    
    // Implement Kernel
    printf("Launching kernel...                 ");    
    startTime(&timer);
    
    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    monte_carlo_simple<<<dimGrid, dimBlock>>>(x_array_d, y_array_d, N, count_array_d);  
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));    
    
    // Copy data from device to host
    printf("Copying data from device to host... ");    
    startTime(&timer);

    cuda_ret = cudaMemcpy(count_array_h, count_array_d, N*sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) printf("Unable to copy count array to host ");

    cudaDeviceSynchronize();        
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));       
    
    // Calculate and verify
    printf("Verifying...                        ");
    startTime(&timer);  
    for (int i = 0; i < N; i++) {
        if (count_array_h[i] == 1) {
            num_inside_circle += 1;
        }
    }
    
    pi_estimate_sc = 4.0 * (float)num_inside_circle / (float)N;    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer)); 
    printf("Pi Estimate: %.6f\n\n", pi_estimate_sc);

    
    // Free memory
    free(x_array_h);
    free(y_array_h);
    free(count_array_h);
    
    cudaFree(x_array_d);
    cudaFree(y_array_d);
    cudaFree(count_array_d);
    
    
    /* Advanced CUDA --------------------------------------------------------*/
    // Method 4
    // Atomic Add
    
    // Init--------------------------------------------------------------------
    printf("--Advanced CUDA--\n");
    printf("Setting up...                       ");
    startTime(&timer);
    
    float pi_estimate_ac;
    num_inside_circle = 0;
    
    // atomicAdd setup
    int *count;
    cudaMallocManaged(&count, 4); // method done by official documentation
    *count = 0;
    
    float *x_array_ac_h, *y_array_ac_h;
    float *x_array_ac_d, *y_array_ac_d;
    
    x_array_ac_h = (float*) malloc( sizeof(float)*N );
    y_array_ac_h = (float*) malloc( sizeof(float)*N );  
    
    // CULPRIT
    for(int i = 0 ; i < N; i++) {
        x_array_ac_h[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        y_array_ac_h[i] = ((float)rand() / (float)RAND_MAX) * 2 - 1;
    }

    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));
    
    // CUDA--------------------------------------------------------------------
    
    // Allocate device memory
    printf("Allocating device memory...         ");    
    startTime(&timer);
    
    cuda_ret = cudaMalloc((void**) &x_array_ac_d, N*sizeof(float));
    if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory for x array ");
    
    cuda_ret = cudaMalloc((void**) &y_array_ac_d, N*sizeof(float));
    if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory for y array ");          
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));
    
    // Copy data from host to device
    printf("Copying data from host to device... ");    
    startTime(&timer);

    cuda_ret = cudaMemcpy(x_array_ac_d, x_array_ac_h, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) printf("Unable to copy x array to device ");    

    cuda_ret = cudaMemcpy(y_array_ac_d, y_array_ac_h, N*sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) printf("Unable to copy y array to device ");

    cudaDeviceSynchronize();    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));    
    
    // Implement Kernel
    printf("Launching kernel...                 ");    
    startTime(&timer);
    
    // dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), 1, 1); already done above
    // dim3 dimBlock(BLOCK_SIZE, 1, 1);               already done above
    monte_carlo_advanced<<<dimGrid, dimBlock>>>(x_array_ac_d, y_array_ac_d, N, count);  
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));    
    
    // Copy data from device to host
    printf("Copying data from device to host... ");    
    startTime(&timer);

    cudaDeviceSynchronize();        
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer));       
    
    // Calculate and verify
    printf("Verifying...                        ");    
    startTime(&timer);    
    
    pi_estimate_ac = 4.0 * (float)*count / (float)N;    
    stopTime(&timer); printf("%f seconds\n", elapsedTime(timer)); 
    printf("Pi Estimate: %.6f\n", pi_estimate_ac);   
    
    // Free memory
    free(x_array_ac_h);
    free(y_array_ac_h);
    
    cudaFree(x_array_ac_d);
    cudaFree(y_array_ac_d);  
    
    return 0;

}
