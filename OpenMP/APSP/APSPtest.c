#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "MatUtil.h"

#include <omp.h>

void PL_APSP_2(int* mat, int N) {
	int all = omp_get_max_threads();
	int per = N / all;
	printf("omp_get_max_threads = %d, per = %d\n", all, per);
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		for (int k = 0; k < N; k++) {
			for (int i = per * (tid); i < per * (tid + 1); i++) {
				for (int j = 0; j < N; j++) {
					int i0 = i * N + j;
					int i1 = i * N + k;
					int i2 = k * N + j;
					if (mat[i1] != -1 && mat[i2] != -1) { 
				        int sum =  (mat[i1] + mat[i2]);
	                    if (mat[i0] == -1 || sum < mat[i0])
					    	mat[i0] = sum;
					}
				}
			}
			#pragma omp barrier
		}
	}
}

void PL_APSP_1(int* mat, int N) {
	printf("omp_get_max_threads = %d\n", omp_get_max_threads());
	#pragma omp parallel shared(mat, N)
	{
		for (int k = 0; k < N; k++) {
			#pragma omp for nowait schedule(static)
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					int i0 = i * N + j;
					int i1 = i * N + k;
					int i2 = k * N + j;
					if (mat[i1] != -1 && mat[i2] != -1) { 
				        int sum =  (mat[i1] + mat[i2]);
	                    if (mat[i0] == -1 || sum < mat[i0])
					    	mat[i0] = sum;
					}
				}
			}
		}
	}
}

void PL_APSP(int* mat, int N) {
	PL_APSP_2(mat, N);
}

long get_time_and_replace(struct timeval *then) {
	// return time from then to now in usec, and assign now to then

	long then_sec = then->tv_sec;
	long then_usec = then->tv_usec;
	gettimeofday(then, NULL);

	long interval = (then->tv_sec - then_sec) * 1000000 + then->tv_usec - then_usec;
	return interval;
}

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Usage: test {N}\n");
		exit(-1);
	}

	// generate a random matrix.
	size_t N = atoi(argv[1]);
	int *mat = (int*)malloc(sizeof(int) * N * N);
	GenMatrix(mat, N);
	printf("Graph Size: %d, Time (usec):\n", N);

	// compute the reference result.
	int *ref = (int*)malloc(sizeof(int) * N * N);
	memcpy(ref, mat, sizeof(int)*N*N);

	struct timeval* time_seq_start = (struct timeval*)malloc(sizeof(struct timeval));
	gettimeofday(time_seq_start, NULL);;
	ST_APSP(ref, N);
	long time_seq_used = get_time_and_replace(time_seq_start);
	printf("Sequential: %8ld\n", time_seq_used);

	// compute your results
	int *result = (int*)malloc(sizeof(int) * N * N);
	memcpy(result, mat, sizeof(int) * N * N);

	//replace by parallel algorithm	
	struct timeval* time_para_start = (struct timeval*)malloc(sizeof(struct timeval));
	gettimeofday(time_para_start, NULL);;
	PL_APSP(result, N);
	long time_para_used = get_time_and_replace(time_para_start);
	printf("Parallel  : %8ld (%.3lf speedup)\n", time_para_used, time_seq_used / (float)time_para_used);

	// compare your result with reference result
	if (CmpArray(result, ref, N * N))
		;
		// printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n");
}
