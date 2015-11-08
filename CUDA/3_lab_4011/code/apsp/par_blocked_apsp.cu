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

#define BLOCK_SIZE 16

__global__ void pass_one(int N, int *mat_device, int start) {
	int i = threadIdx.y;
	int j = threadIdx.x;

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i = start + i;
	int mat_j = start + j;
	int mat_ij = mat_i * N + mat_j;
	block[i][j] = mat_device[mat_ij];
	// __syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		__syncthreads();

		int dik = block[i][k];
		int dkj = block[k][j];
		if (dik != -1 && dkj != -1) {
			int d = dik + dkj;
			int& dij = block[i][j];
			if (dij == -1 || dij > d)
				dij = d;
		}
	}

	mat_device[mat_ij] = block[i][j];
}

inline void do_one(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE, 1);
	pass_one<<<dimBlock, dimGrid>>>(N, mat_device, start);

}

__global__ void pass_two(int N, int *mat_device, int start) {
	int i = threadIdx.y;
	int j = threadIdx.x;

	__shared__ int prime[BLOCK_SIZE][BLOCK_SIZE];
	int prime_mat_i = start + i;
	int prime_mat_j = start + j;
	int prime_mat_ij = prime_mat_i * N + prime_mat_j;
	prime[i][j] = mat_device[prime_mat_ij];
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
	
	block[i][j] = mat_device[mat_ij];
	// __syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		int dik = prime[i][k];
		int dkj = prime[k][j];
		if (dik != -1 && dkj != -1) {
			int d = dik + dkj;
			int& dij = block[i][j];
			if (dij == -1 || dij > d)
				dij = d;
		}
		// __syncthreads();
	}

	mat_device[mat_ij] = block[i][j];
}

inline void do_two(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(2, n_block - 1, 1); // 0: horizontal ---, 1: vertical |||
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE, 1);
	pass_two<<<dimBlock, dimGrid>>>(N, mat_device, start);

}

__global__ void pass_tre(int N, int *mat_device, int start) {
	int i = threadIdx.y;
	int j = threadIdx.x;

	__shared__ int hor[BLOCK_SIZE][BLOCK_SIZE];
	int hor_mat_i = start + i;
	int hor_mat_j = start + j;
	if (hor_mat_j >= start)
		hor_mat_j += BLOCK_SIZE;
	int hor_mat_ij = hor_mat_i * BLOCK_SIZE + hor_mat_j;
	hor[i][j] = mat_device[hor_mat_ij];
	// __syncthreads();

	__shared__ int ver[BLOCK_SIZE][BLOCK_SIZE];
	int ver_mat_i = start + i;
	if (ver_mat_i >= start)
		ver_mat_i += BLOCK_SIZE;
	int ver_mat_j = start + j;
	int ver_mat_ij = ver_mat_i * BLOCK_SIZE + ver_mat_j;
	ver[i][j] = mat_device[ver_mat_ij];
	__syncthreads();

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i = blockIdx.y * BLOCK_SIZE + i;
	int mat_j = blockIdx.x * BLOCK_SIZE + j;
	if (mat_i >= start)
		mat_i += BLOCK_SIZE;
	if (mat_j >= start)
		mat_j += BLOCK_SIZE;
	int mat_ij = mat_i * N + mat_j;
	block[i][j] = mat_device[mat_ij];
	// __syncthreads();

	for (int k = 0; k < BLOCK_SIZE; k++) {
		int dik = hor[i][k];
		int dkj = ver[k][j];
		if (dik != -1 && dkj != -1) {
			int d = dik + dkj;
			int& dij = block[i][j];
			if (dij == -1 || dij > d)
				dij = d;
		}
		// __syncthreads();
	}

	mat_device[mat_ij] = block[i][j];
}

inline void do_tre(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(n_block - 1, n_block - 1, 1); // 0: horizontal ---, 1: vertical |||
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE, 1);
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
