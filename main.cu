/******************************************************************************
MPP Final - Summer 2025
Monte Carlo Optimization - Main
Jake Webb
******************************************************************************/

// Estimate pi using the monte carlo method

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    // Init Main---------------------------------------------------------------
    Timer timer;
    startTime(&timer);
    cudaError_t cuda_ret;
    
    int N;
    if(argc == 1) {
        N = 1000000;
    } else if(argc == 2) {
        N = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./prefix-scan        # Input of size 1,000,000 is used"
           "\n    Usage: ./prefix-scan <m>    # Input of size m is used"
           "\n");
        exit(0);
    }
    
    /* Simple Serial Code ---------------------------------------------------*/
    // Generate an X and Y vector of random numbers
    // Loop through the vectors, calculate
    
    /* Advanced Serial Code -------------------------------------------------*/
    // Optimization 1
    // Better memory efficiency, uses single points instead of vectors
    
    // Init Serial-------------------------------------------------------------
    float pi_estimate; 
    int num_inside_circle = 0; 
    
    float point_x;
    float point_y;
    
    float calc_time; // how long it takes to finish the whole calculation
    
    // Compute-----------------------------------------------------------------
    
    // Main Loop
    for(int i = 0; i < N; i++) {
        point_x = ((float)rand() / (float)RAND_MAX) * 2 - 1; // Calculate from -1 to 1
        point_y = ((float)rand() / (float)RAND_MAX) * 2 - 1;
        
        if (point_x*point_x + point_y*point_y < 1) {
            num_inside_circle += 1;
        }
    }

    // Estimate Pi
    pi_estimate = 4.0 * (float)num_inside_circle / (float)N;
    
    // Time
    stopTime(&timer); 
    calc_time = elapsedTime(timer);
    
    // Output------------------------------------------------------------------
    printf("--Serial--\n");
    printf("Input Size: %d\n", N);
    printf("Pi Estimate: %.8f\n", pi_estimate);
    printf("Time to Calculate: %f s\n\n", calc_time);
    
    
    /* Simple CUDA ----------------------------------------------------------*/
    // Optimization 2
    // Threading
    
    
    /* Complex CUDA ---------------------------------------------------------*/
    // Optimization 3
    // AtomicAdd()
    
    
    
    
    
    
    return 0;
}