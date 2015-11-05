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
struct timeval timer_setup;
long time_setup;

// helper function for struct timeval
long get_time_interval(struct timeval t_from, struct timeval t_to) {
	// return time from t_from to t_to in usec
	return (t_from.tv_sec - t_to.tv_sec) * 1000000 + t_from.tv_usec - t_to.tv_usec;
}
long get_time_and_replace(struct timeval *then) {
	// return time from then to now in usec, and assign now to then
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

void get_rows_distribution(int mat_size, int my_rank, int total_rank, int* mainly_have_rows, int* rows_in_charge) {
	// rows_in_charge = mat_size / P;
	// 0 = my_rank * mat_size / P;
	// rows_in_charge = (my_rank + 1) * mat_size / P;
	*mainly_have_rows = mat_size / total_rank;
	if ((*mainly_have_rows) * total_rank != mat_size) {
		++(*mainly_have_rows);
		if (my_rank + 1 == total_rank)
			*rows_in_charge = mat_size % (*mainly_have_rows);
		else
			*rows_in_charge = *mainly_have_rows;
	}
	else
		*rows_in_charge = *mainly_have_rows;
}

void get_distribution_array(int mat_size, int mainly_have_rows, int total_rank, int* rows_distribution, int* rows_offset, int* ele_distribution, int* ele_offset) {
	for (int i = 0; i < total_rank; i++)
		rows_distribution[i] = mainly_have_rows;
	if (mat_size % mainly_have_rows != 0)
		rows_distribution[total_rank - 1] = mat_size % mainly_have_rows;
	rows_offset[0] = 0;
	for (int i = 1; i < total_rank; i++)
		rows_offset[i] = rows_offset[i - 1] + rows_distribution[i - 1];
	for (int i = 0; i < total_rank; i++) {
		ele_distribution[i] = mat_size * rows_distribution[i];
		ele_offset[i] = mat_size * rows_offset[i];
	}
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
	
	int* mat;
	int* ref;
	int* result;
	if (my_rank == 0) {
		printf("Input size: %ld\n", mat_size);
		printf("Total process: %d\n", total_rank);

		// compute answer in sequential, and copy data for parallel
		prepare_data(&mat, &ref, &result, mat_size);

		// start the timer for parallel algorithm
		gettimeofday(&timer_parallel, NULL);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// start the timer for setup
	if (my_rank == 0)
		gettimeofday(&timer_setup, NULL);

	// know the set of rows I am working on according to my_rank 
	int mainly_have_rows;
	int rows_in_charge;
	get_rows_distribution(mat_size, my_rank, total_rank, &mainly_have_rows, &rows_in_charge);

	int* my_rows = (int*)malloc(sizeof(int) * mat_size * rows_in_charge); // rows the current process have
	int* k_to_j = (int*)malloc(sizeof(int) * mat_size); // the vertical (column)
	int* rows_distribution = (int*)malloc(sizeof(int) * total_rank); // how many rows does process i have
	int* rows_offset = (int*)malloc(sizeof(int) * total_rank); // how many rows does process (0..i-1) have
	int* ele_distribution = (int*)malloc(sizeof(int) * total_rank); //how many numbers does process i have
	int* ele_offset = (int*)malloc(sizeof(int) * total_rank); // how many numbers does process (0..i-1) have
	// fill in the arrays above, for used in communication
	get_distribution_array(mat_size, mainly_have_rows, total_rank, rows_distribution, rows_offset, ele_distribution, ele_offset);

	// stop the timer for setup
	if (my_rank == 0)
		time_setup += get_time_and_replace(&timer_setup);

	// start the timer for communication
	if (my_rank == 0)
		gettimeofday(&timer_comm, NULL);
	// divide the matrix for each process
	// send rows to each process using scatter, sendbuf:*result, recvbuf:*my_rows
	// MPI_Scatter(
	//     void* send_data,
	//     int send_count,
	//     MPI_Datatype send_datatype,
	//     void* recv_data,
	//     int recv_count,
	//     MPI_Datatype recv_datatype,
	//     int root,
	//     MPI_Comm communicator)
	// int MPI_Scatterv(
	// 	void *sendbuf, 
	// 	int *sendcounts, 
	// 	int *displs,
	// 	MPI_Datatype sendtype, 
	// 	void *recvbuf, 
	// 	int recvcount,
	// 	MPI_Datatype recvtype, 
	// 	int root, 
	// 	MPI_Comm comm)
	int sendrecvcount = mat_size * rows_in_charge;
	MPI_Scatterv(
		result, 
		ele_distribution, 
		ele_offset, 
		MPI_INT, 
		my_rows, 
		sendrecvcount, 
		MPI_INT, 
		0, 
		MPI_COMM_WORLD);
	// stop the timer for communication
	if (my_rank == 0)
		time_comm += get_time_and_replace(&timer_comm);

	preprocess_graph(my_rows, rows_in_charge, mat_size);

	for (int k = 0; k < mat_size; k++) {
		// start the timer for communication
		if (my_rank == 0)
			gettimeofday(&timer_comm, NULL);

		// broadcast k-th row to other process if I am the owner
		int owner_of_k_row = k / mainly_have_rows;
		if (my_rank == owner_of_k_row)
			memcpy(k_to_j, my_rows + mat_size * (k % mainly_have_rows), sizeof(int) * mat_size);
		// MPI_Bcast(
		//     void* data,
		//     int count,
		//     MPI_Datatype datatype,
		//     int root,
		//     MPI_Comm communicator)
		MPI_Bcast(k_to_j, mat_size, MPI_INT, owner_of_k_row, MPI_COMM_WORLD);
		if (my_rank == 0)
			time_comm += get_time_and_replace(&timer_comm);

		for (int i = 0; i < rows_in_charge; i++)
			for (int j = 0; j < mat_size; j++) {
				int cur_index = i * mat_size + j;
				if (my_rows[cur_index] > my_rows[i * mat_size + k] + k_to_j[j])
					my_rows[cur_index] = my_rows[i * mat_size + k] + k_to_j[j];
			}
	}

	// start the timer for communication
	if (my_rank == 0)
		gettimeofday(&timer_comm, NULL);

	// collect result to process 0
	// MPI_Gather(
	//     void* send_data,
	//     int send_count,
	//     MPI_Datatype send_datatype,
	//     void* recv_data,
	//     int recv_count,
	//     MPI_Datatype recv_datatype,
	//     int root,
	//     MPI_Comm communicator)
	// int MPI_Gatherv(
	// 	const void *sendbuf, 
	// 	int sendcount, 
	// 	MPI_Datatype sendtype,
	// 	void *recvbuf, 
	// 	const int *recvcounts, 
	// 	const int *displs,
	// 	MPI_Datatype recvtype, 
	// 	int root, 
	// 	MPI_Comm comm)
	MPI_Gatherv(
		my_rows,
		sendrecvcount,
		MPI_INT,
		result,
		ele_distribution,
		ele_offset,
		MPI_INT,
		0,
		MPI_COMM_WORLD);

	// stop the timer for communication
	if (my_rank == 0)
		time_comm += get_time_and_replace(&timer_comm);

	if (my_rank == 0) {
		//stop the timer for parallel algorithm
		time_used_parallel = get_time_and_replace(&timer_parallel);
		printf("Time used (parallel  ): %8ld usecs\n", time_used_parallel);
		printf("Time used (parallel  ) setup: %6ld usecs (%2.3lf%%) \n", time_setup, time_setup / (double)time_used_parallel * 100);
		printf("Time used (parallel  ) comm : %6ld usecs (%2.3lf%%) \n", time_comm, time_comm / (double)time_used_parallel * 100);
		printf("Speed up (sequential / parallel): %.3lf\n", time_used_sequential / (double)time_used_parallel);

		// compare the result with reference result
		if(CmpArray(result, ref, mat_size * mat_size))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");
	}

    // Finalize the MPI environment.
    MPI_Finalize();
}
