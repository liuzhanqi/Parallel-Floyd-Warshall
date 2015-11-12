#define BLOCK_SIZE 128

__global__ void apspKernel(int N, int k, int *global_mat) {
    const int ij = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int i = ij / N;
    const int ik = i * N + k;
    const int kj = ij + (k - i) * N;
    const int dik = global_mat[ik];
    const int dkj = global_mat[kj];
    if (dik != -1 && dkj != -1) {
        const int sum = dik + dkj;
        int& dij = global_mat[ij];
        if (dij == -1 || dij > sum) {
            dij = sum;
        }
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