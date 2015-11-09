#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "apsp.h"

void gen_apsp(int N, int *mat) {
	for (int i = 0; i < N * N; i++)
		mat[i] = rand() % 32 - 1;
	// for (int i = 0; i < N * N; i++)
	// 	if (mat[i] < 0)
	// 		mat[i] = 33;
	// for (int i = 0; i < N; i++)
	// 	mat[i * N + i] = 0;

	ref_N = N;
	ref = (int*)malloc(sizeof(int) * N * N);
	for (int i = 0; i < N * N; i++)
		ref[i] = mat[i];
}

void seq_apsp() {
	for (int k = 0; k < ref_N; k++) {
		for (int i = 0; i < ref_N; i++) {
			for (int j = 0; j < ref_N; j++) {
				int i0 = i * ref_N + j;
				int i1 = i * ref_N + k;
				int i2 = k * ref_N + j;
				// int sum = mat[i1] + mat[i2];
				// if (sum < mat[i0])
				// 	mat[i0] = sum;
				if (ref[i1] != -1 && ref[i2] != -1) { 
			        int sum = (ref[i1] + ref[i2]);
                    if (ref[i0] == -1 || sum < ref[i0])
 					    ref[i0] = sum;
				}
			}
		}
	}
}

bool check_apsp(int *mat) {
	for (int i = 0; i < ref_N * ref_N; i ++) {
		if (mat[i] != ref[i]) {
			printf("ERROR: mat[%d] = %d, ref[%d] = %d\n", i, mat[i], i, ref[i]);
			return false;
		}
	}
	return true;
}
