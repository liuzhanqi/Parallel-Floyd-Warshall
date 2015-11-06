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

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void testKernel(float *g_idata, float *g_odata) {
    // shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}

// void example() {
// 	unsigned int num_threads = 32;
//     unsigned int mem_size = sizeof(float) * num_threads;

//     // allocate host memory
//     float *h_idata = mat;

//     // initalize the memory
//     for (unsigned int i = 0; i < num_threads; ++i)
//         h_idata[i] = (float)i;

//     // allocate device memory
//     float *d_idata;
//     checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
//     // copy host memory to device
//     checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

//     // allocate device memory for result
//     float *d_odata;
//     checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

//     // setup execution parameters
//     dim3 grid(1, 1, 1);
//     dim3 threads(num_threads, 1, 1);

//     // execute the kernel
//     testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);

//     // check if kernel execution generated and error
//     getLastCudaError("Kernel execution failed");

//     // allocate mem for the result on host side
//     float *h_odata = (float *) malloc(mem_size);
//     // copy result from device to host
//     checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads, cudaMemcpyDeviceToHost));

//     // compute reference solution
//     float *reference = (float *) malloc(mem_size);
//     computeGold(reference, h_idata, num_threads);

//     // custom output handling when no regression test running
//     // in this case check if the result is equivalent to the expected solution
//     bool bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);

//     // cleanup memory
//     free(h_idata);
//     free(h_odata);
//     free(reference);
//     checkCudaErrors(cudaFree(d_idata));
//     checkCudaErrors(cudaFree(d_odata));

//     if (bTestResult)
//     	printf("SUCCESS\n");
//     else
//     	printf("FAILED - RESULT WRONG\n");	
// }

__global__ void apspKernel(int N, int k, int *g_idata, int *g_odata) {
    // access thread id
    const unsigned int tid = threadIdx.x;
    // access block id
    const unsigned int bid = blockIdx.x;
    // access number of threads in this block
    const unsigned int bdim = blockDim.x;

    const unsigned int i = (bid * bdim + tid)/N;
    const unsigned int j = (bid * bdim + tid)%N;

    g_odata[i*N+j] = min(g_idata[i*N+j], g_idata[i*N+k]+g_idata[k*N+j]);
}

void par_apsp(int N, int *mat) {
    //copy mat from host to device memory d_mat
    int* d_mat;
    int* d_mat_out;
    int size = sizeof(int) * N * N;
    cudaMalloc((void**) &d_mat, size);
    cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &d_mat_out, size);

    for (int k=0; k<N; k++) {
        apspKernel<<<ceil(N*N/256), 256>>>(N, k, d_mat, d_mat_out);
        cudaMemcpy(d_mat, d_mat_out, size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);
}


