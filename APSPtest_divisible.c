#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <mpi.h>

#include "MatUtil.h"

int my_rank;
int total_rank;

struct timeval timer_sequential;
long time_used_sequential;
struct timeval timer_parallel;
long time_used_parallel;

struct timeval timer_comm;
long time_comm;

// helper function for struct timeval
long get_time_interval(struct timeval t_from, struct timeval t_to) {
	return (t_from.tv_sec - t_to.tv_sec) * 1000000 + t_from.tv_usec - t_to.tv_usec;
}
long get_time_and_replace(struct timeval *then) {
	struct timeval* now = (struct timeval*)malloc(sizeof(struct timeval));
	gettimeofday(now, NULL);

	long interval = get_time_interval(*now, *then);
	then = now;
	return interval;
}

void prepare_data(int** mat, int** ref, int** result, int mat_size) {
	//generate a random matrix.
	*mat = (int*)malloc(sizeof(int) * mat_size * mat_size);
	GenMatrix(*mat, mat_size);

	//compute the reference result.
	*ref = (int*)malloc(sizeof(int) * mat_size * mat_size);
	memcpy(*ref, *mat, sizeof(int) * mat_size * mat_size);
	// start sequential timer
	gettimeofday(&timer_sequential, NULL);
	ST_APSP(*ref, mat_size);
	// stop sequential timer
	time_used_sequential = get_time_and_replace(&timer_sequential);
	printf("Time used (sequential): %8ld usecs\n", time_used_sequential);

	// compute your result
	*result = (int*)malloc(sizeof(int) * mat_size * mat_size);
	memcpy(*result, *mat, sizeof(int) * mat_size * mat_size);
}

void preprocess_graph(int* my_rows, int rows, int cols) {
	// set -1 edge (disconnected) to INT_MAX / 2
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++) {
			int cur_index = i * cols + j;
			if (my_rows[cur_index] < 0)
				my_rows[cur_index] = INT_MAX / 2;
		}	
}

int main(int argc, char **argv) {
	MPI_Init(NULL, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &total_rank);

	if(argc != 2) {
		printf("Usage: test {mat_size}\n");
		exit(-1);
	}
	size_t mat_size = atoi(argv[1]);

	if (mat_size % total_rank) {
		for (int i = 0; i < 20; i++)
			printf("~");
		printf("\n");
		printf("Using MPT_*, only support divisible number of vertex...\n");
		for (int i = 0; i < 20; i++)
			printf("~");
		printf("\n");
		exit(-1);
	}
	
	int* mat;
	int* ref;
	int* result;
	if (my_rank == 0) {
		for (int i = 0; i < 20; i++)
			printf("~");
		printf("\n");
		printf("Using MPT_*, only support divisible number of vertex...\n");
		printf("Input size: %ld\n", mat_size);
		printf("Total process: %d\n", total_rank);

		prepare_data(&mat, &ref, &result, mat_size);

		// start the timer
		gettimeofday(&timer_parallel, NULL);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// know the set of rows I am working on according to my_rank 
	int rows_in_charge = mat_size / total_rank;

	// should this be included in the timer?
	int* my_rows = (int*)malloc(sizeof(int) * mat_size * rows_in_charge); //rows the current process have
	int* k_to_j = (int*)malloc(sizeof(int) * mat_size); // the vertical (column)

	if (my_rank == 0)
		gettimeofday(&timer_comm, NULL);
	// divide the matrix for each process
	// send rows to each process using scatter, sendbuf:*result, recvbuf:*my_rows
	int sendrecvcount = mat_size * rows_in_charge;
	MPI_Scatter(
		result, 
		sendrecvcount, 
		MPI_INT, 
		my_rows, 
		sendrecvcount, 
		MPI_INT, 
		0, 
		MPI_COMM_WORLD);
	if (my_rank == 0)
		time_comm += get_time_and_replace(&timer_comm);

	// preprocess_graph(my_rows, rows_in_charge, mat_size);

	for (int k = 0; k < mat_size; k++) {
		if (my_rank == 0)
			gettimeofday(&timer_comm, NULL);
		// broadcast k-th row to other process if I am the owner
		int owner_of_k_row = k / rows_in_charge;
		if (my_rank == owner_of_k_row)
			memcpy(k_to_j, my_rows + mat_size * (k % rows_in_charge), sizeof(int) * mat_size);
		MPI_Bcast(k_to_j, mat_size, MPI_INT, owner_of_k_row, MPI_COMM_WORLD);
		if (my_rank == 0)
			time_comm += get_time_and_replace(&timer_comm);

		for (int i = 0; i < rows_in_charge; i++) {
			for (int j = 0; j < mat_size; j++) {
				int ij = i * mat_size + j;
				int ik = i * mat_size + k;
				// if (my_rows[ij] > ikj)
				// 	my_rows[ij] = ikj;
				if (my_rows[ik] != -1 && k_to_j[j] != -1) {
					int ikj = my_rows[ik] + k_to_j[j];
					if (my_rows[ij] == -1 || my_rows[ij] > ikj)
						my_rows[ij] = ikj;
				}
			}
		}
	}

	if (my_rank == 0)
		gettimeofday(&timer_comm, NULL);
	// collect result to process 0
	MPI_Gather(
		my_rows,
		sendrecvcount,
		MPI_INT,
		result,
		sendrecvcount,
		MPI_INT,
		0,
		MPI_COMM_WORLD);
	if (my_rank == 0)
		time_comm += get_time_and_replace(&timer_comm);

	if (my_rank == 0) {
		//stop the timer
		time_used_parallel = get_time_and_replace(&timer_parallel);
		printf("Time used (parallel  ): %8ld usecs\n", time_used_parallel);
		printf("Time used (parallel  ) comm : %6ld usecs (%2.3lf%%) \n", time_comm, time_comm / (double)time_used_parallel * 100);
		printf("Speed up (sequential / parallel): %.3lf\n", time_used_sequential / (double)time_used_parallel);

		//compare your result with reference result
		if(CmpArray(result, ref, mat_size * mat_size))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");
		for (int i = 0; i < 20; i++)
			printf("~");
		printf("\n");
	}

    // Finalize the MPI environment.
    MPI_Finalize();
}
