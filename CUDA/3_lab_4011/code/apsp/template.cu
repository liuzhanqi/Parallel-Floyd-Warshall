// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples

// cly: includes, apsp
#include "apsp.h"

#if (APSP_VERSION == 1)
#include "par_apsp.h"
#else
#include "par_blocked_apsp.h"
#include "configure.h"
#endif

#ifndef REPEAT
#define REPEAT 1
#endif



void run_apsp(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: test {N - #node}\n");
        return ;
    }

    size_t N = atoi(argv[1]);
    int *mat;
#ifdef UNIFIED_MEMORY
    cudaMallocManaged(&mat, sizeof(int) * N * N);
#else
    mat = (int*)malloc(sizeof(int) * N * N);
#endif
    gen_apsp(N, mat);

    double avg_par_time = 0;
    for (int i = 1; i <= REPEAT; i++) {

        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

#if (APSP_VERSION == 1)
        par_apsp_blocked_processing(N, mat);
#else
        par_blocked_apsp(N, mat);
#endif

        sdkStopTimer(&timer);
        long par_time = sdkGetTimerValue(&timer);    
        sdkDeleteTimer(&timer);

        avg_par_time += (par_time - avg_par_time) / i;
    }

#ifdef RUN_SEQUENTIAL
    double avg_seq_time = 0;
    for (int i = 1; i <= REPEAT; i++) {

        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        seq_apsp();
        
        sdkStopTimer(&timer);
        long seq_time = sdkGetTimerValue(&timer);
        sdkDeleteTimer(&timer);

        avg_seq_time += (seq_time - avg_seq_time) / i;
    }
#endif

#ifdef UNIFIED_MEMORY
    printf("use Unified Memory\n");
#endif
    printf("Graph size: %3d, repeat %d times\n", (int)N, REPEAT);
    #if (BLOCK_SIZE && THREAD_SIZE)
    printf("Processing time: %10.3lf (ms) parallel (block %dx%d thread size %d\n", avg_par_time, BLOCK_SIZE, BLOCK_SIZE, THREAD_SIZE);
    #else
    printf("Processing time: %10.3lf (ms) parallel\n", avg_par_time);
    #endif
#ifdef RUN_SEQUENTIAL
    printf("Processing time: %10.3lf (ms) sequential\n", avg_seq_time);
    printf("Speed-up: %5.3lf\n", avg_seq_time / avg_par_time);
#endif

#ifdef RUN_SEQUENTIAL
    check_apsp(mat);
#endif

#ifdef UNIFIED_MEMORY
    cudaFree(mat);
#endif
    cudaDeviceReset();

}



__global__ void wake_up_gpu() {}

int main(int argc, char **argv) {
    wake_up_gpu<<<dim3(32, 32, 1), dim3(32, 32 ,1)>>>();
    run_apsp(argc, argv);
}
