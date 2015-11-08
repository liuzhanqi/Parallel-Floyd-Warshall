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
// #include "par_apsp.h"
#include "par_blocked_apsp.h"

void run_apsp(int argc, char **argv);

int main(int argc, char **argv) {
    run_apsp(argc, argv);
}

void run_apsp(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: test {N - #node}\n");
        exit(-1);
    }

    size_t N = atoi(argv[1]);
    int *mat = (int*)malloc(sizeof(int) * N * N);
    gen_apsp(N, mat);

    float par_time;
    float seq_time;
    {
        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        par_blocked_apsp(N, mat);

        sdkStopTimer(&timer);
        par_time = sdkGetTimerValue(&timer);
        #if (BLOCK_SIZE && THREAD_SIZE)
        printf("Processing time: %10.3f (ms) parallel (block %dx%d thread %dx%d\n", par_time, BLOCK_SIZE, BLOCK_SIZE, THREAD_SIZE, THREAD_SIZE);
        #else
        printf("Processing time: %10.3f (ms) parallel\n", par_time);
        #endif
        sdkDeleteTimer(&timer);        
    }

    {
        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        seq_apsp();
        
        sdkStopTimer(&timer);
        seq_time = sdkGetTimerValue(&timer);
        printf("Processing time: %10.3f (ms) sequential\n", seq_time);
        sdkDeleteTimer(&timer);
    }

    const char *description = "GPU function with XXX (secret) optimization.";
    printf("Description: %s\n", description);
    printf("Speed-up: %5.3f\n", seq_time / par_time);
    check_apsp(mat);

    cudaDeviceReset();
}