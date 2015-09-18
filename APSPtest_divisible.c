#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <sys/time.h>
#include <mpi.h>

#include "MatUtil.h"

#define min(x, y) ((x) < (y) ? (x) : (y))

// #define DEBUG_MSG

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

#ifdef DEBUG_MSG
void shout_name() {
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int total_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &total_rank);

	printf("Hi, I am process %d...\n", my_rank, total_rank);
}
void show_dd_matrix(int* arr, int row, int col) {
	printf("matrix %2dx%d\n", row, col);
	for (int i = 0; i < row; i++) {
		printf("row %2d:", i);
		for (int j = 0; j < col; j++)
			printf(" %2d", arr[i * col + j]);
		printf("\n");
	}
}
void show_arr(int* arr, int size) {
	printf("arr[%d]:", size);
	for (int i = 0; i < size; i++)
		printf(" %d", arr[i]);
	printf("\n");
}
#endif

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

#ifdef DEBUG_MSG
	shout_name();
	printf("original matrix: \n");
	show_dd_matrix(*result, mat_size, mat_size);
	printf("\n");
#endif
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

	// start timer for setup
	if (my_rank == 0)
		gettimeofday(&timer_setup, NULL);

	// know the set of rows I am working on according to my_rank 
	int rows_in_charge = mat_size / total_rank;
#ifdef DEBUG_MSG
	shout_name();
	printf("I charges %d rows.\n", rows_in_charge);
#endif

	// should this be included in the timer?
	int* my_rows = (int*)malloc(sizeof(int) * mat_size * rows_in_charge); //rows the current process have
	int* k_to_j = (int*)malloc(sizeof(int) * mat_size); // the vertical (column)

	if (my_rank == 0)
		time_setup += get_time_and_replace(&timer_setup);

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
#ifdef DEBUG_MSG
	shout_name();
	printf("my_rows: \n");
	show_dd_matrix(my_rows, rows_in_charge, mat_size);
#endif

	preprocess_graph(my_rows, rows_in_charge, mat_size);

	for (int k = 0; k < mat_size; k++) {
		if (my_rank == 0)
			gettimeofday(&timer_comm, NULL);
		// broadcast k-th row to other process if I am the owner
		int owner_of_k_row = k / rows_in_charge;
		if (my_rank == owner_of_k_row)
			memcpy(k_to_j, my_rows + mat_size * (k % rows_in_charge), sizeof(int) * mat_size);
		// MPI_Bcast(
		//     void* data,
		//     int count,
		//     MPI_Datatype datatype,
		//     int root,
		//     MPI_Comm communicator)
		MPI_Bcast(k_to_j, mat_size, MPI_INT, owner_of_k_row, MPI_COMM_WORLD);
		if (my_rank == 0)
			time_comm += get_time_and_replace(&timer_comm);

#ifdef DEBUG_MSG
		if (my_rank == 1)
			MPI_Barrier(MPI_COMM_WORLD);

		printf("iteration %d process %d my_rows:\n", k, my_rank);
		show_dd_matrix(my_rows, rows_in_charge, mat_size);
		printf("i_to_k: [");
		for (int i = 0; i < mat_size; i++)
			printf("%2d,", i_to_k[i]);
		printf("]\n");
		printf("k_to_j: [");
		for (int i = 0; i < mat_size; i++)
			printf("%2d,", k_to_j[i]);
		printf("]\n");

		if (my_rank != 1)
			MPI_Barrier(MPI_COMM_WORLD);
#endif

		for (int i = 0; i < rows_in_charge; i++) {
			for (int j = 0; j < mat_size; j++) {
				int cur_index = i * mat_size + j;
				if (my_rows[cur_index] > my_rows[i * mat_size + k] + k_to_j[j])
					my_rows[cur_index] = my_rows[i * mat_size + k] + k_to_j[j];
			}
		}

#ifdef DEBUG_MSG
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gather(
			my_rows,
			sendrecvcount,
			MPI_INT,
			result,
			sendrecvcount,
			MPI_INT,
			0,
			MPI_COMM_WORLD);
		if (my_rank == 0) {
			printf("\ncheckpoint iteration %d result: \n", k);
			show_dd_matrix(result, mat_size, mat_size);
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	}

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
		printf("Time used (parallel  ) setup: %6ld usecs (%2.3lf%%) \n", time_setup, time_setup / (double)time_used_parallel * 100);
		printf("Time used (parallel  ) comm : %6ld usecs (%2.3lf%%) \n", time_comm, time_comm / (double)time_used_parallel * 100);
		printf("Speed up (sequential / parallel): %.3lf\n", time_used_sequential / (double)time_used_parallel);
#ifdef DEBUG_MSG
		printf("Correct Answer: \n");
		show_dd_matrix(ref, mat_size, mat_size);
		printf("\n");
		printf("FINAL Result: \n");
		show_dd_matrix(result, mat_size, mat_size);
		printf("\n");
#endif
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
