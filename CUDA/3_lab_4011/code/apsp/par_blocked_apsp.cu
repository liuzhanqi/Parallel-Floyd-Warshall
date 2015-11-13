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
#include "configure.h"

// 
// reference
// 
// http://docs.nvidia.com/cuda/profiler-users-guide/index.html#compute-command-line-profiler-overview
// http://devblogs.nvidia.com/parallelforall/maxwell-most-advanced-cuda-gpu-ever-made/
// http://stackoverflow.com/questions/6563261/how-to-use-coalesced-memory-access
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/
// http://stackoverflow.com/questions/21797582/cuda-6-simplest-sample-segmentation-fault
// https://devtalk.nvidia.com/default/topic/517828/speed-up-initialization-of-cuda-about-how-to-set-the-device-code-translation-cache/
// http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/
// http://jasonjuang.blogspot.sg/2014/02/what-is-bank-conflict-in-cuda.html
// 
// Katz, Gary J., and Joseph T. Kider Jr. 
// "All-pairs shortest-paths for large graphs on the GPU." 
// Proceedings of the 23rd ACM SIGGRAPH/EUROGRAPHICS symposium on Graphics hardware. 
// Eurographics Association, 2008.
// 

__device__ inline void assign(int N, int des[][BLOCK_SIZE], int i, int j, int *src, int mat_ij) {
	for (int t = 0; t < THREAD_SIZE; t++)
#ifdef THREAD_DO_VERTICAL
		des[i + t][j] = src[mat_ij + t * N];
#else
		des[i][j + t] = src[mat_ij + t];
#endif
}

__device__ inline void assign(int N, int *des, int mat_ij, int src[][BLOCK_SIZE], int i, int j) {
	for (int t = 0; t < THREAD_SIZE; t++)
#ifdef THREAD_DO_VERTICAL
		des[mat_ij + t * N] = src[i + t][j];
#else
		des[mat_ij + t] = src[i][j + t];
#endif
}



__global__ void pass_one(int N, int *mat_device, int start) {
	int i = threadIdx.y * THREAD_SIZE;
	int j = threadIdx.x;

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i = start + i;
	int mat_j = start + j;
	int mat_ij = mat_i * N + mat_j;
	for (int t = 0; t < THREAD_SIZE; t++)
		block[i + t][j] = mat_device[mat_ij + t * N];

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
#ifdef THREAD_DO_VERTICAL
	int i = threadIdx.y * THREAD_SIZE;
	int j = threadIdx.x;
#else
	int i = threadIdx.y;
	int j = threadIdx.x * THREAD_SIZE;
#endif

	__shared__ int hor[BLOCK_SIZE][BLOCK_SIZE];
	int hor_mat_i = start + i;
	int hor_mat_j = start + j;
	if (hor_mat_j >= start)
		hor_mat_j += BLOCK_SIZE;
	int hor_mat_ij = hor_mat_i * BLOCK_SIZE + hor_mat_j;
	assign(N, hor, i, j, mat_device, hor_mat_ij);
	// for (int t = 0; t < THREAD_SIZE; t++)
	// 	hor[i + t][j] = mat_device[hor_mat_ij + t * N];

	__shared__ int ver[BLOCK_SIZE][BLOCK_SIZE];
	int ver_mat_i = start + i;
	if (ver_mat_i >= start)
		ver_mat_i += BLOCK_SIZE;
	int ver_mat_j = start + j;
	int ver_mat_ij = ver_mat_i * BLOCK_SIZE + ver_mat_j;
	assign(N, ver, i, j, mat_device, ver_mat_ij);
	// for (int t = 0; t < THREAD_SIZE; t++)
	// 	ver[i + t][j] = mat_device[ver_mat_ij + t * N];
	__syncthreads();

	__shared__ int block[BLOCK_SIZE][BLOCK_SIZE];
	int mat_i = blockIdx.y * BLOCK_SIZE + i;
	int mat_j = blockIdx.x * BLOCK_SIZE + j;
	if (mat_i >= start)
		mat_i += BLOCK_SIZE;
	if (mat_j >= start)
		mat_j += BLOCK_SIZE;
	int mat_ij = mat_i * N + mat_j;
	assign(N, block, i, j, mat_device, mat_ij);
	// for (int t = 0; t < THREAD_SIZE; t++)
	// 	block[i + t][j] = mat_device[mat_ij + t * N];

	for (int k = 0; k < BLOCK_SIZE; k++) {
		for (int t = 0; t < THREAD_SIZE; t++) {
#ifdef THREAD_DO_VERTICAL
			int dik = hor[i + t][k];
			int dkj = ver[k][j];
#else
			int dik = hor[i][k];
			int dkj = ver[k][j + t];
#endif
			if (dik != -1 && dkj != -1) {
				int d = dik + dkj;
#ifdef THREAD_DO_VERTICAL
				int& dij = block[i + t][j];
#else
				int& dij = block[i][j + t];
#endif
				if (dij == -1 || dij > d)
					dij = d;
			}
		}
	}

	assign(N, mat_device, mat_ij, block, i, j);
	// for (int t = 0; t < THREAD_SIZE; t++)
	// 	mat_device[mat_ij + t * N] = block[i + t][j];
}

inline void do_tre(int N, int *mat_device, int n_block, int start) {

	dim3 dimBlock(n_block - 1, n_block - 1, 1);
#ifdef THREAD_DO_VERTICAL
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE / THREAD_SIZE, 1);
#else
	dim3 dimGrid(BLOCK_SIZE / THREAD_SIZE, BLOCK_SIZE, 1);
#endif
	pass_tre<<<dimBlock, dimGrid>>>(N, mat_device, start);

}

void par_blocked_apsp(int N, int *mat) {
    
    int *mat_device;
#ifdef UNIFIED_MEMORY
    mat_device = mat;
#else
    size_t size = N * N * sizeof(int);
    cudaMalloc(&mat_device, size);
    cudaMemcpy(mat_device, mat, size, cudaMemcpyHostToDevice);
#endif

    int n_block = N / BLOCK_SIZE;
    for (int i = 0; i < n_block; i++) {
    	int start = i * BLOCK_SIZE;
        do_one(N, mat_device, n_block, start);
        do_two(N, mat_device, n_block, start);
        do_tre(N, mat_device, n_block, start);
    }

#ifdef UNIFIED_MEMORY
    cudaDeviceSynchronize();
#else
    cudaMemcpy(mat, mat_device, size, cudaMemcpyDeviceToHost);
    cudaFree(mat_device);
#endif

}





// __global__ void process_type_1_tile_kernel(int N, int *mat_device, int start)
// {
// 	int i = threadIdx.y * THREAD_SIZE;
// 	int j = threadIdx.x;

// 	__shared__ int tile[TILE_SIZE][TILE_SIZE];
// 	int mat_i = start + i;
// 	int mat_j = start + j;
// 	int mat_ij = mat_i * N + mat_j;
// 	for (int t = 0; t < THREAD_SIZE; t++) // load tile into shared memory
// 		tile[i + t][j] = mat_device[mat_ij + t * N];

// 	for (int k = 0; k < TILE_SIZE; k++) {
// 		for (int t = 0; t < THREAD_SIZE; t++) {
// 			__syncthreads();

// 			int dik = tile[i + t][k];
// 			int dkj = tile[k][j];
// 			if (dik != -1 && dkj != -1) {
// 				int d = dik + dkj;
// 				int& dij = tile[i + t][j];
// 				if (dij == -1 || dij > d)
// 					dij = d;
// 			}	
// 		}
// 	}

// 	for (int t = 0; t < THREAD_SIZE; t++) // store tile back to global memory
// 		tile[i + t][j] = mat_device[mat_ij + t * N];
// }

// inline void process_type_1_tile(int N, int *mat_device, int n_tile, int start)
// {
// 	dim3 dimBlock(1, 1, 1);
// 	dim3 dimGrid(TILE_SIZE, TILE_SIZE / THREAD_SIZE, 1);
// 	process_type_1_tile_kernel<<<dimBlock, dimGrid>>>(N, mat_device, start);
// }

// __global__ void process_type_2_tile_kernel(int N, int *mat_device, int start)
// {
// 	int i = threadIdx.y * THREAD_SIZE;
// 	int j = threadIdx.x;

// 	__shared__ int prime[TILE_SIZE][TILE_SIZE];
// 	int prime_mat_i = start + i;
// 	int prime_mat_j = start + j;
// 	int prime_mat_ij = prime_mat_i * N + prime_mat_j;
// 	for (int t = 0; t < THREAD_SIZE; t++) // load tile into shared memory
// 		prime[i + t][j] = mat_device[prime_mat_ij + t * N];
// 	__syncthreads();

// 	__shared__ int tile[TILE_SIZE][TILE_SIZE];
// 	int mat_i;
// 	int mat_j;
// 	if (blockIdx.x == 0) {
// 		mat_i = start + i;
// 		mat_j = blockIdx.y * TILE_SIZE + j;
// 		if (mat_j >= start)
// 			mat_j += TILE_SIZE;
// 	}
// 	else {
// 		mat_i = blockIdx.y * TILE_SIZE + i;
// 		if (mat_i >= start)
// 			mat_i += TILE_SIZE;
// 		mat_j = start + j;
// 	}
// 	int mat_ij = mat_i * N + mat_j;
	
// 	for (int t = 0; t < THREAD_SIZE; t++) // load tile into shared memory
// 		tile[i + t][j] = mat_device[mat_ij + t * N];

// 	for (int k = 0; k < TILE_SIZE; k++) {
// 		for (int t = 0; t < THREAD_SIZE; t++) {
// 			int dik = prime[i + t][k];
// 			int dkj = prime[k][j];
// 			if (dik != -1 && dkj != -1) {
// 				int d = dik + dkj;
// 				int& dij = tile[i + t][j];
// 				if (dij == -1 || dij > d)
// 					dij = d;
// 			}
// 		}
// 	}

// 	for (int t = 0; t < THREAD_SIZE; t++) // store tile back to global memory
// 		mat_device[mat_ij + t * N] = tile[i + t][j];
// }

// inline void process_type_2_tile(int N, int *mat_device, int n_tile, int start)
// {
// 	dim3 dimBlock(2, n_tile - 1, 1); // 0: horizontal ---, 1: vertical |||
// 	dim3 dimGrid(TILE_SIZE, TILE_SIZE / THREAD_SIZE, 1);
// 	process_type_2_tile_kernel<<<dimBlock, dimGrid>>>(N, mat_device, start);
// }

// __global__ void process_type_3_tile_kernel(int N, int *mat_device, int start)
// {
// #ifdef THREAD_DO_VERTICAL
// 	int i = threadIdx.y * THREAD_SIZE;
// 	int j = threadIdx.x;
// #else
// 	int i = threadIdx.y;
// 	int j = threadIdx.x * THREAD_SIZE;
// #endif

// 	__shared__ int hor[TILE_SIZE][TILE_SIZE];
// 	int hor_mat_i = start + i;
// 	int hor_mat_j = start + j;
// 	if (hor_mat_j >= start)
// 		hor_mat_j += TILE_SIZE;
// 	int hor_mat_ij = hor_mat_i * TILE_SIZE + hor_mat_j;
// 	for (int t = 0; t < THREAD_SIZE; t++) // load tile into shared memory
// #ifdef THREAD_DO_VERTICAL
// 		hor[i + t][j] = mat_device[hor_mat_ij + t * N];
// #else
// 		hor[i][j + t] = mat_device[hor_mat_ij + t];
// #endif

// 	__shared__ int ver[TILE_SIZE][TILE_SIZE];
// 	int ver_mat_i = start + i;
// 	if (ver_mat_i >= start)
// 		ver_mat_i += TILE_SIZE;
// 	int ver_mat_j = start + j;
// 	int ver_mat_ij = ver_mat_i * TILE_SIZE + ver_mat_j;
// 	for (int t = 0; t < THREAD_SIZE; t++) // load tile into shared memory
// #ifdef THREAD_DO_VERTICAL
// 		hor[i + t][j] = mat_device[ver_mat_ij + t * N];
// #else
// 		hor[i][j + t] = mat_device[ver_mat_ij + t];
// #endif
// 	__syncthreads();

// 	__shared__ int tile[TILE_SIZE][TILE_SIZE];
// 	int mat_i = blockIdx.y * TILE_SIZE + i;
// 	int mat_j = blockIdx.x * TILE_SIZE + j;
// 	if (mat_i >= start)
// 		mat_i += TILE_SIZE;
// 	if (mat_j >= start)
// 		mat_j += TILE_SIZE;
// 	int mat_ij = mat_i * N + mat_j;
// 	for (int t = 0; t < THREAD_SIZE; t++) // load tile into shared memory
// #ifdef THREAD_DO_VERTICAL
// 		tile[i + t][j] = mat_device[mat_device + t * N];
// #else
// 		tile[i][j + t] = mat_device[mat_device + t];
// #endif

// 	for (int k = 0; k < TILE_SIZE; k++) {
// 		for (int t = 0; t < THREAD_SIZE; t++) {
// #ifdef THREAD_DO_VERTICAL
// 			int dik = hor[i + t][k];
// 			int dkj = ver[k][j];
// #else
// 			int dik = hor[i][k];
// 			int dkj = ver[k][j + t];
// #endif
// 			if (dik != -1 && dkj != -1) {
// 				int d = dik + dkj;
// #ifdef THREAD_DO_VERTICAL
// 				int& dij = tile[i + t][j];
// #else
// 				int& dij = tile[i][j + t];
// #endif
// 				if (dij == -1 || dij > d)
// 					dij = d;
// 			}
// 		}
// 	}

// 	for (int t = 0; t < THREAD_SIZE; t++) // store tile back to global memory
// #ifdef THREAD_DO_VERTICAL
// 		mat_device[mat_ij + t * N] = tile[i + t][j];
// #else
// 		mat_device[mat_ij + t] = tile[i][j + t];
// #endif
// }

// inline void process_type_3_tile(int N, int *mat_device, int n_tile, int start)
// {
// 	dim3 dimBlock(n_tile - 1, n_tile - 1, 1);
// #ifdef THREAD_DO_VERTICAL
// 	dim3 dimGrid(TILE_SIZE, TILE_SIZE / THREAD_SIZE, 1);
// #else
// 	dim3 dimGrid(TILE_SIZE / THREAD_SIZE, TILE_SIZE, 1);
// #endif
// 	process_type_3_tile_kernel<<<dimBlock, dimGrid>>>(N, mat_device, start);
// }

// void par_blocked_apsp(int N, int *mat)
// {
//     int *mat_device;
//     size_t size = N * N * sizeof(int);
//     cudaMalloc(&mat_device, size);
//     cudaMemcpy(mat_device, mat, size, cudaMemcpyHostToDevice);

//     int n_tile = N / TILE_SIZE;
//     for (int i = 0; i < n_tile; i++) {
//     	int start = i * TILE_SIZE;
//         process_type_1_tile(N, mat_device, n_tile, start);
//         process_type_2_tile(N, mat_device, n_tile, start);
//         process_type_3_tile(N, mat_device, n_tile, start);
//     }

//     cudaMemcpy(mat, mat_device, size, cudaMemcpyDeviceToHost);
//     cudaFree(mat_device);
// }
