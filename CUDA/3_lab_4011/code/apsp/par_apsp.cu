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

#define BLOCK_SIZE 8
#define THREAD_SIZE 8

#define INF 599999999

__global__ void kernel_phase_one(const unsigned int block, 
                                 const unsigned int N, 
                                 int * const d) {
    int i;
    int newPath;

    const int VIRTUAL_BLOCK_SIZE = BLOCK_SIZE * THREAD_SIZE;

    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;
    
    const int v1 = VIRTUAL_BLOCK_SIZE * block + ty;
    const int v2 = VIRTUAL_BLOCK_SIZE * block + tx;  

    const int cell = v1 * N + v2;

    __shared__ int primary_d[VIRTUAL_BLOCK_SIZE][VIRTUAL_BLOCK_SIZE];
    // __shared__ int primary_p[VIRTUAL_BLOCK_SIZE][VIRTUAL_BLOCK_SIZE];

    if (v1 < N && v2 < N) primary_d[ty][tx] = d[cell];
    else primary_d[ty][tx] = INF;

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

    for (i=0; i<VIRTUAL_BLOCK_SIZE; i++) {
        if (primary_d[ty][i] != -1 && primary_d[i][tx] != -1) {
            newPath = primary_d[ty][i] + primary_d[i][tx];
            if (newPath < primary_d[ty][tx] || primary_d[ty][tx] == -1) primary_d[ty][tx] = newPath;
        }
    }

    if (v1 < N && v2 < N) {
        d[cell] = primary_d[ty][tx];
    }
}


__global__ void kernel_phase_two(const unsigned int block, 
                                 const unsigned int N, 
                                 int * const d) {
    if (blockIdx.x == block) return;
    const int VIRTUAL_BLOCK_SIZE = BLOCK_SIZE * THREAD_SIZE;

    int i;
    int newPath;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int v1 = VIRTUAL_BLOCK_SIZE * block + ty;
    int v2 = VIRTUAL_BLOCK_SIZE * block + tx;
    
    __shared__ int primary_d[VIRTUAL_BLOCK_SIZE][VIRTUAL_BLOCK_SIZE];
    __shared__ int current_d[VIRTUAL_BLOCK_SIZE][VIRTUAL_BLOCK_SIZE];

    const int cell_primary = v1 * N + v2;
    if (v1 < N && v2 < N) primary_d[ty][tx] = d[cell_primary];
    else primary_d[ty][tx] = INF;
    
    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        v1 = VIRTUAL_BLOCK_SIZE * block + ty;
        v2 = VIRTUAL_BLOCK_SIZE * blockIdx.x + tx;
    }
    // Load j-aligned singly dependent blocks
    else  {
        v1 = VIRTUAL_BLOCK_SIZE * blockIdx.x + ty;
        v2 = VIRTUAL_BLOCK_SIZE * block + tx;
    }
    
    const int cell_current = v1 * N + v2;
    if (v1 < N && v2 < N) current_d[ty][tx] = d[cell_current];
    else current_d[ty][tx] = INF;

    // Synchronize to make sure the all value are loaded in block
    __syncthreads();

    // Compute i-aligned singly dependent blocks
    if (blockIdx.y == 0)
    {
        for (i=0; i<VIRTUAL_BLOCK_SIZE; i++) {
            if (primary_d[ty][i] != -1  && current_d[i][tx] != -1) {
                newPath = primary_d[ty][i] + current_d[i][tx];
                
                if (newPath < current_d[ty][tx] || current_d[ty][tx] == -1) current_d[ty][tx] = newPath;
            }
        }
    }
    // Compute j-aligned singly dependent blocks
    else {
        for (i=0; i<VIRTUAL_BLOCK_SIZE; i++) {
            if (current_d[ty][i] != -1 && primary_d[i][tx] != -1) {
                newPath = current_d[ty][i] + primary_d[i][tx];
            
            if (newPath < current_d[ty][tx] || current_d[ty][tx] == -1) current_d[ty][tx] = newPath;
            }
        }
    }

    if (v1 < N && v2 < N) d[cell_current] = current_d[ty][tx];
}

__global__ void kernel_phase_three(unsigned int block, 
                                   const unsigned int N, 
                                   int * const d) {
    if (blockIdx.x == block || blockIdx.y == block) return;

    int i, j, k;    
    int newPath;
    int path;

    const int tx = threadIdx.x * THREAD_SIZE;
    const int ty = threadIdx.y * THREAD_SIZE;

    const int v1 = blockDim.y * blockIdx.y * THREAD_SIZE + ty;
    const int v2 = blockDim.x * blockIdx.x * THREAD_SIZE + tx;

    int idx, idy;
    
    __shared__ int primaryRow_d[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
    __shared__ int primaryCol_d[BLOCK_SIZE * THREAD_SIZE][BLOCK_SIZE * THREAD_SIZE];
    
    int v1Row = BLOCK_SIZE * block * THREAD_SIZE + ty;
    int v2Col = BLOCK_SIZE * block * THREAD_SIZE + tx;

    for (i=0; i<THREAD_SIZE; i++) {
        for(j=0; j<THREAD_SIZE; j++) {
            idx = tx + j;
            idy = ty + i;
        
            if (v1Row + i < N && v2 + j < N) {
                block = (v1Row + i) * N + v2 + j;
                primaryRow_d[idy][idx] = d[block];
            }
            else {
                primaryRow_d[idy][idx] = INF;
            }
        
            if (v1 + i  < N && v2Col + j < N) {
                block = (v1 + i) * N + v2Col + j;
                primaryCol_d[idy][idx] = d[block];
            }
            else {
                primaryCol_d[idy][idx] = INF;
            }
        }
    }
    
     // Synchronize to make sure the all value are loaded in virtual block
    __syncthreads();

    for (i=0; i<THREAD_SIZE; i++) {
        for (j=0; j<THREAD_SIZE; j++) {
            if (v1 + i < N && v2 + j < N) {
                block = (v1 + i) * N + v2 + j;
                        
                path = d[block];
                
                idy = ty + i;
                idx = tx + j;

                for (k=0; k<BLOCK_SIZE * THREAD_SIZE; k++) {
                    if (primaryCol_d[idy][k] != -1 && primaryRow_d[k][idx] != -1) {
                        newPath = primaryCol_d[idy][k] + primaryRow_d[k][idx];
                        if (path > newPath || path == -1) {
                            path = newPath;
                        }
                    }
                }
                d[block] = path;
            }
        }
    }
}

void par_apsp_blocked_processing(int N, int *mat) {
    //copy mat from host to device memory d_mat
    int* d_mat;
    int size = sizeof(int) * N * N;
    cudaMalloc((void**) &d_mat, size);
    cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);

    const int VIRTUAL_BLOCK_SIZE = BLOCK_SIZE * THREAD_SIZE;
    // Initialize the grid and block dimensions here
    dim3 dimGridP1(1, 1, 1);
    dim3 dimGridP2((N - 1) / VIRTUAL_BLOCK_SIZE + 1, 2 , 1);
    dim3 dimGridP3((N - 1) / VIRTUAL_BLOCK_SIZE + 1, (N - 1) / VIRTUAL_BLOCK_SIZE + 1, 1);

    dim3 dimBlockP1(VIRTUAL_BLOCK_SIZE, VIRTUAL_BLOCK_SIZE, 1);
    dim3 dimBlockP2(VIRTUAL_BLOCK_SIZE, VIRTUAL_BLOCK_SIZE, 1);
    dim3 dimBlockP3(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    int numOfBlock = (N - 1) / VIRTUAL_BLOCK_SIZE;

    for (int block = 0; block <= numOfBlock; block++) {
        kernel_phase_one<<<1, dimBlockP1>>>(block, N, d_mat);
        kernel_phase_two<<<dimGridP2, dimBlockP2>>>(block, N, d_mat);
        kernel_phase_three<<<dimGridP3, dimBlockP3>>>(block, N, d_mat);       
    }

    cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);
}
