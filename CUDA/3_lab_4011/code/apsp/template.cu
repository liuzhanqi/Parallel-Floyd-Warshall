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

#include "configure.h"

#if (APSP_VERSION == 1)
#include "par_apsp.h"
#elif (APSP_VERSION == 2)
#include "par_apsp_naive.h"
#else
#include "par_blocked_apsp.h"
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



    printf("Graph size: %3d, Repeat %d times\n", (int)N, REPEAT);
#if defined(RUN_PARALLEL)
    #if (APSP_VERSION != 2)
        #if defined(BLOCK_SIZE) && defined(THREAD_SIZE)
            const char* conf = "Run Parallel... block %dx%d thread size %d %s %s ";
        #else
            const char* conf = "Run Parallel... %s %s\n";
        #endif
        #ifdef THREAD_DO_VERTICAL
            const char* direction = "|||";
        #else
            const char* direction = "---";
        #endif

        #ifdef UNIFIED_MEMORY
            const char* unified = "use Unified Memory";
        #else
            const char* unified = "";
        #endif
        #if defined(BLOCK_SIZE) && defined(THREAD_SIZE)
            printf(conf, BLOCK_SIZE, BLOCK_SIZE, THREAD_SIZE, direction, unified);
        #else
            printf(conf, BLOCK_SIZE, direction, unified);
        #endif
    #else
        printf("Run Parallel... naive block size (128?) ");
    #endif
#endif



#ifdef RUN_PARALLEL
    double avg_par_time = 0;
    for (int i = 1; i <= REPEAT; i++) {

        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

#if (APSP_VERSION == 1)
        par_apsp_blocked_processing(N, mat);
#elif (APSP_VERSION == 2)
        par_apsp(N, mat);
#else
        par_blocked_apsp(N, mat);
#endif

        sdkStopTimer(&timer);
        long par_time = sdkGetTimerValue(&timer);    
        sdkDeleteTimer(&timer);

        avg_par_time += (par_time - avg_par_time) / i;
    }

    printf("Processing Time: \t%5.3lf ms\n", avg_par_time);
#endif

#ifdef RUN_SEQUENTIAL
    printf("Run Sequential... ");
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

    printf("Processing Time: \t\t\t\t%10.3lf ms\n", avg_seq_time);
#endif



#if defined(RUN_PARALLEL) && defined(RUN_SEQUENTIAL)
    printf("Speed-up: x%5.3lf\n", avg_seq_time / avg_par_time);
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
