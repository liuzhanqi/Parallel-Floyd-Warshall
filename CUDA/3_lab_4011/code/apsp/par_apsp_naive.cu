#define BLOCK_SIZE 256

__global__ void apspKernel(const int N, const int k, int *global_mat) {
    const int ij = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int i = ij / N;
    const int ik = i * N + k;
    const int kj = ij + (k - i) * N;
    const int dik = global_mat[ik];
    const int dkj = global_mat[kj];

    if (dik != -1 && dkj != -1) {
        const int sum = dik + dkj;
        int& dij = global_mat[ij];
        if (dij == -1 || dij > sum)
            dij = sum;
    }
}

void par_apsp(int N, int *mat) {
    int *global_mat, size = sizeof(int) * N * N;
    cudaMalloc(&global_mat, size);
    cudaMemcpy(global_mat, mat, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(N * N / BLOCK_SIZE, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    for (int k = 0; k < N; k++)
        apspKernel<<<dimGrid, dimBlock>>>(N, k, global_mat);

    cudaMemcpy(mat, global_mat, size, cudaMemcpyDeviceToHost);
}

// __global__ void apspKernel(int N, int k, int *g_idata, int *g_odata) {
//     // access thread id
//     const unsigned int tid = threadIdx.x;
//     // access block id
//     const unsigned int bid = blockIdx.x;
//     // access number of threads in this block
//     const unsigned int bdim = blockDim.x;

//     const unsigned int i = (bid * bdim + tid)/N;
//     const unsigned int j = (bid * bdim + tid)%N;

//     if (g_idata[i*N+k] == -1 || g_idata[k*N+j] == -1) g_odata[i*N+j] = g_idata[i*N+j];
//     else if (g_idata[i*N+j] == -1) g_odata[i*N+j] = g_idata[i*N+k]+g_idata[k*N+j];
//     else g_odata[i*N+j] = min(g_idata[i*N+j], g_idata[i*N+k]+g_idata[k*N+j]);
// }

// void par_apsp(int N, int *mat) {
//     //copy mat from host to device memory d_mat
//     int* d_mat;
//     int* d_mat_out;
//     int size = sizeof(int) * N * N;
//     cudaMalloc((void**) &d_mat, size);
//     cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
//     //allocate matrix to hold temporary result of each iteration to avoid race condition.
//     cudaMalloc((void**) &d_mat_out, size);

//     for (int k = 0; k < N; k++) {
//         apspKernel<<<ceil(N*N/256), 256>>>(N, k, d_mat, d_mat_out);
//         //copy the temporary result back to the matrix
//         cudaMemcpy(d_mat, d_mat_out, size, cudaMemcpyDeviceToDevice);
//     }

//     cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);
// }

// __global__ void apspKernel(int N, int k, int *g_data) {
//     const int tid = threadIdx.x;
//     const int bid = blockIdx.x;
//     const int bdim = blockDim.x;

//     const int ij = bid * bdim + tid;
    
//     const int i = ij / N;
//     const int j = ij % N;

//     const int ik = i * N + k;
//     const int kj = k * N + j;
//     const int dik = g_data[ik];
//     const int dkj = g_data[kj];
    
//     if (dik != -1 && dkj != -1) {
//         const int sum = dik + dkj;
//         int& dij = g_data[ij];
//         if (dij == -1 || dij > sum)
//             dij = sum;
//     }
// }

// void par_apsp(int N, int *mat) {
//     int* d_mat;
//     int size = sizeof(int) * N * N;
//     cudaMalloc((void**) &d_mat, size);
//     cudaMemcpy(d_mat, mat, size, cudaMemcpyHostToDevice);
    
//     for (int k = 0; k < N; k++)
//         apspKernel<<<ceil(N*N/256), 256>>>(N, k, d_mat);

//     cudaMemcpy(mat, d_mat, size, cudaMemcpyDeviceToHost);
// }