#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>

#include "MatUtil.h"

#define min(x, y) ((x) < (y) ? (x) : (y))

// #define DEBUG_MSG

struct timeval tv1,tv2;
long time_used_sequential;
long time_used_parallel;

int main(int argc, char **argv) {
	if(argc != 2) {
		printf("Usage: test {mat_size}\n");
		exit(-1);
	}
	size_t mat_size = atoi(argv[1]);

	int* mat = (int*)malloc(sizeof(int) * mat_size * mat_size);
	GenMatrix(mat, mat_size);

	// start sequential timer
	gettimeofday(&tv1, NULL);
	ST_APSP(mat, mat_size);
	// stop sequential timer
	gettimeofday(&tv2, NULL);
	time_used_sequential = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
	printf("Time used (sequential): %8ld usecs\n", time_used_sequential); 	
}
