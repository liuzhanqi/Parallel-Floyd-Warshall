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

#include "par_blocked_apsp.h"

// 
// reference
// 
// http://docs.nvidia.com/cuda/profiler-users-guide/index.html#compute-command-line-profiler-overview
// http://devblogs.nvidia.com/parallelforall/maxwell-most-advanced-cuda-gpu-ever-made/
// http://stackoverflow.com/questions/6563261/how-to-use-coalesced-memory-access
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/
// http://stackoverflow.com/questions/21797582/cuda-6-simplest-sample-segmentation-fault
// 
// Katz, Gary J., and Joseph T. Kider Jr. 
// "All-pairs shortest-paths for large graphs on the GPU." 
// Proceedings of the 23rd ACM SIGGRAPH/EUROGRAPHICS symposium on Graphics hardware. 
// Eurographics Association, 2008.
// 


__global__ void pass_one(int N, int *mat_device, int start) {
	int i = threadIdx.y * THREAD_SIZE;
	int j = threadIdx.x;

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i = start + i;
	int mat_j = start + j;
	int mat_ij = mat_i * N + mat_j;
	for (int t = 0; t < THREAD_SIZE; t++)
		block[i + t][j] = mat_device[mat_ij + t * N];

	// __syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		for (int t = 0; t < THREAD_SIZE; t++) {
			__syncthreads();

			int dik = block[i + t][k];
			int dkj = block[k][j];
			if (dik != -1 && dkj != -1) {
				int d = dik + dkj;
				int& dij = block[i + t][j];
				if (dij == -1 || dij > d)
					dij = d;
			}	
		}
	}

	for (int t = 0; t < THREAD_SIZE; t++)
		block[i + t][j] = mat_device[mat_ij + t * N];
}

inline void do_one(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE, 1);
	pass_one<<<dimBlock, dimGrid>>>(N, mat_device, start);

}

__global__ void pass_two(int N, int *mat_device, int start) {
	int i = threadIdx.y * THREAD_SIZE;
	int j = threadIdx.x;

	__shared__ int prime[BLOCK_SIZE][BLOCK_SIZE];
	int prime_mat_i = start + i;
	int prime_mat_j = start + j;
	int prime_mat_ij = prime_mat_i * N + prime_mat_j;
	for (int t = 0; t < THREAD_SIZE; t++)
		prime[i + t][j] = mat_device[prime_mat_ij + t * N];
	__syncthreads();

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i;
	int mat_j;
	if (blockIdx.x == 0) {
		mat_i = start + i;
		mat_j = blockIdx.y * BLOCK_SIZE + j;
		if (mat_j >= start)
			mat_j += BLOCK_SIZE;
	}
	else {
		mat_i = blockIdx.y * BLOCK_SIZE + i;
		if (mat_i >= start)
			mat_i += BLOCK_SIZE;
		mat_j = start + j;
	}
	int mat_ij = mat_i * N + mat_j;
	
	for (int t = 0; t < THREAD_SIZE; t++)
		block[i + t][j] = mat_device[mat_ij + t * N];
	// __syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		for (int t = 0; t < THREAD_SIZE; t++) {
			int dik = prime[i + t][k];
			int dkj = prime[k][j];
			if (dik != -1 && dkj != -1) {
				int d = dik + dkj;
				int& dij = block[i + t][j];
				if (dij == -1 || dij > d)
					dij = d;
			}
			// __syncthreads();	
		}
	}

	for (int t = 0; t < THREAD_SIZE; t++)
		mat_device[mat_ij + t * N] = block[i + t][j];
}

inline void do_two(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(2, n_block - 1, 1); // 0: horizontal ---, 1: vertical |||
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE, 1);
	pass_two<<<dimBlock, dimGrid>>>(N, mat_device, start);

}

__global__ void pass_tre(int N, int *mat_device, int start) {
	int i = threadIdx.y * THREAD_SIZE;
	int j = threadIdx.x;

	__shared__ int hor[BLOCK_SIZE][BLOCK_SIZE];
	int hor_mat_i = start + i;
	int hor_mat_j = start + j;
	if (hor_mat_j >= start)
		hor_mat_j += BLOCK_SIZE;
	int hor_mat_ij = hor_mat_i * BLOCK_SIZE + hor_mat_j;
	// for (int t = 0; t < THREAD_SIZE; t++)
	// 	hor[i + t][j] = mat_device[hor_mat_ij + t * N];
	// __syncthreads();

	__shared__ int ver[BLOCK_SIZE][BLOCK_SIZE];
	int ver_mat_i = start + i;
	if (ver_mat_i >= start)
		ver_mat_i += BLOCK_SIZE;
	int ver_mat_j = start + j;
	int ver_mat_ij = ver_mat_i * BLOCK_SIZE + ver_mat_j;
	for (int t = 0; t < THREAD_SIZE; t++) {
		hor[i + t][j] = mat_device[hor_mat_ij + t * N];
		ver[i + t][j] = mat_device[ver_mat_ij + t * N];
	}
	__syncthreads();

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i = blockIdx.y * BLOCK_SIZE + i;
	int mat_j = blockIdx.x * BLOCK_SIZE + j;
	if (mat_i >= start)
		mat_i += BLOCK_SIZE;
	if (mat_j >= start)
		mat_j += BLOCK_SIZE;
	int mat_ij = mat_i * N + mat_j;
	for (int t = 0; t < THREAD_SIZE; t++)
		block[i + t][j] = mat_device[mat_ij + t * N];
	// __syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		for (int t = 0; t < THREAD_SIZE; t++) {
			int dik = hor[i + t][k];
			int dkj = ver[k][j];
			if (dik != -1 && dkj != -1) {
				int d = dik + dkj;
				int& dij = block[i + t][j];
				if (dij == -1 || dij > d)
					dij = d;
			}
			// __syncthreads();	
		}
	}

	for (int t = 0; t < THREAD_SIZE; t++)
		mat_device[mat_ij + t * N] = block[i + t][j];
}

inline void do_tre(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(n_block - 1, n_block - 1, 1);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE, 1);
	pass_tre<<<dimBlock, dimGrid>>>(N, mat_device, start);

}

void par_blocked_apsp(int N, int *mat) {
    
    int *mat_device;
    size_t size = N * N * sizeof(int);
    cudaMalloc(&mat_device, size);
    cudaMemcpy(mat_device, mat, size, cudaMemcpyHostToDevice);

    int n_block = N / BLOCK_SIZE;
    for (int i = 0; i < n_block; i++) {
    	int start = i * BLOCK_SIZE;
        do_one(N, mat_device, n_block, start);
        do_two(N, mat_device, n_block, start);
        do_tre(N, mat_device, n_block, start);
    }

    cudaMemcpy(mat, mat_device, size, cudaMemcpyDeviceToHost);
    cudaFree(mat_device);

}
