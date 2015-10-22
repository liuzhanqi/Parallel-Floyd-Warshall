#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "MatUtil.h"

#include <omp.h>

// #define APPROACH -1

#if APPROACH == 0

int per, tid, i0, i1, i2, sum, i, j, k;
#pragma omp threadprivate(tid, i0, i1, i2, sum, i, j)

void PL_APSP_X(int* mat, int N, int num_thread) {
	per = N / num_thread;
	k = 0;
	#pragma omp parallel shared(per, k)
	{
		tid = omp_get_thread_num();
		while (k < N) {
			for (i = per * tid; i < per * (tid + 1); ++i) {
				for (j = 0; j < N; ++j) {
					i0 = i * N + j;
					i1 = i * N + k;
					i2 = k * N + j;
					sum = mat[i1] + mat[i2];
					if (sum < mat[i0])
						mat[i0] = sum;
				}
			}
			#pragma omp master
			{
				k++;
			}
			#pragma omp barrier
		}
	}
}

#elif APPROACH == 1

int per, tid, i0, i1, i2, sum, i, j, k;
#pragma omp threadprivate(tid, i0, i1, i2, sum, i, j, k)

void PL_APSP_X(int* mat, int N, int num_thread) {
	per = N / num_thread;
	#pragma omp parallel shared(per)
	{
		tid = omp_get_thread_num();
		for (k = 0; k < N; k++) {
			for (i = per * tid; i < per * (tid + 1); ++i) {
				for (j = 0; j < N; ++j) {
					i0 = i * N + j;
					i1 = i * N + k;
					i2 = k * N + j;
					sum = mat[i1] + mat[i2];
					if (sum < mat[i0])
						mat[i0] = sum;
				}
			}
			#pragma omp barrier
		}
	}
}

#elif APPROACH == 2

void PL_APSP_X(int* mat, int N, int num_thread) {
	int per = N / num_thread, tid, i0, i1, i2, sum, i, j, k = 0;
	#pragma omp parallel shared(per, k) private(tid, i0, i1, i2, sum, i, j)
	{
		tid = omp_get_thread_num();
		while (k < N) {
			for (i = per * tid; i < per * (tid + 1); ++i) {
				for (j = 0; j < N; ++j) {
					i0 = i * N + j;
					i1 = i * N + k;
					i2 = k * N + j;
					sum = mat[i1] + mat[i2];
					if (sum < mat[i0])
						mat[i0] = sum;
				}
			}
			#pragma omp master
			{
				k++;
			}
			#pragma omp barrier
		}
	}
}

#elif APPROACH == 3

void PL_APSP_X(int* mat, int N, int num_thread) {
	int per = N / num_thread, tid, i0, i1, i2, sum, i, j, k;
	#pragma omp parallel shared(per) private(tid, i0, i1, i2, sum, i, j, k)
	{
		tid = omp_get_thread_num();
		for (k = 0; k < N; k++) {
			for (i = per * tid; i < per * (tid + 1); ++i) {
				for (j = 0; j < N; ++j) {
					i0 = i * N + j;
					i1 = i * N + k;
					i2 = k * N + j;
					sum = mat[i1] + mat[i2];
					if (sum < mat[i0])
						mat[i0] = sum;
				}
			}
			#pragma omp barrier
		}
	}
}

#else

void PL_APSP_X(int* mat, int N, int num_thread) {
	printf("[INVALID VALUE FOR APPROACH]\n");
}

#endif

void PL_APSP(int* mat, int N, int num_thread) {
	printf("omp_get_num_threads = %d\n", num_thread);
	omp_set_num_threads(num_thread);
	PL_APSP_X(mat, N, num_thread);
}

long get_time(struct timeval *then) {
	// return time from [then] to [now] in usec
	long then_sec = then->tv_sec;
	long then_usec = then->tv_usec;
	gettimeofday(then, NULL);

	return (then->tv_sec - then_sec) * 1000000 + then->tv_usec - then_usec;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		printf("Usage: test {N - #node} {M - #thread}\n");
		exit(-1);
	}

	// generate a random matrix.
	size_t N = atoi(argv[1]);
	int num_thread = atoi(argv[2]);
	int *mat = (int*)malloc(sizeof(int) * N * N);
	GenMatrix(mat, N);

	printf("Graph Size: %d, #thread: %d, Time (usec):\n", N, num_thread);

	// compute the reference result.
	int *ref = (int*)malloc(sizeof(int) * N * N);
	memcpy(ref, mat, sizeof(int)*N*N);

	struct timeval time_seq_start;
	gettimeofday(&time_seq_start, NULL);;
	ST_APSP(ref, N);
	long time_seq_used = get_time(&time_seq_start);
	printf("Sequential: %8ld\n", time_seq_used);

	// compute your results
	int *result = (int*)malloc(sizeof(int) * N * N);
	memcpy(result, mat, sizeof(int) * N * N);

	// replace by parallel algorithm
	for (int t = 2; t <= num_thread; t += 2) {
		struct timeval time_para_start;
		gettimeofday(&time_para_start, NULL);
		PL_APSP(result, N, t);
		long time_para_used = get_time(&time_para_start);
		printf("Parallel(%d): %8ld (%.3lf speedup)\n", APPROACH, time_para_used, time_seq_used / (float)time_para_used);		
	}

	// compare your result with reference result
	if (CmpArray(result, ref, N * N))
		;
		// printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n");
	printf("\n");
}
