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
struct timeval tv1,tv2;
long time_used_sequential;
long time_used_parallel;

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
		for (int j = 0; j < col; j++) {
			printf(" %2d", arr[i * col + j]);
		}
		printf("\n");
	}
}
void show_arr(int* arr, int size) {
	printf("arr[%d]:", size);
	for (int i = 0; i < size; i++)
		printf(" %d", arr[i]);
	printf("\n");
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
	
	if (my_rank == 0) {
		printf("Input size: %ld\n", mat_size);
		printf("Total process: %d\n", total_rank);
	}

	int* mat;
	int* ref;
	int* result;
	if (my_rank == 0)  {
		//generate a random matrix.
		mat = (int*)malloc(sizeof(int) * mat_size * mat_size);
		GenMatrix(mat, mat_size);

		//compute the reference result.
		ref = (int*)malloc(sizeof(int) * mat_size * mat_size);
		memcpy(ref, mat, sizeof(int) * mat_size * mat_size);
		// TODO start sequential timer
		gettimeofday(&tv1, NULL);
		ST_APSP(ref, mat_size);
		// TODO stop sequential timer
		gettimeofday(&tv2, NULL);
		time_used_sequential = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
		printf("Time used (sequential): %8ld usecs\n", time_used_sequential); 

		// compute your result
		result = (int*)malloc(sizeof(int) * mat_size * mat_size);
		memcpy(result, mat, sizeof(int) * mat_size * mat_size);

#ifdef DEBUG_MSG
		shout_name();
		printf("original matrix: \n");
		show_dd_matrix(result, mat_size, mat_size);
		printf("\n");
#endif
	}

	if (my_rank == 0) {
		// TODO start the timer
		gettimeofday(&tv1, NULL);	
	}


	// know the set of rows I am working on according to my_rank 
	// numRow = mat_size/P;
	// rowStart = my_rank*mat_size/P;
	// rowEnd = (my_rank+1)*mat_size/P;
	int rows_in_charge = mat_size / total_rank;
	if (mat_size % total_rank != 0)
		rows_in_charge = my_rank == total_rank - 1 ? mat_size % (rows_in_charge + 1) : rows_in_charge + 1;
	int row_start = 0;
	int row_end = rows_in_charge;
// #ifdef DEBUG_MSG
// 	shout_name();
// 	printf("I charges %d rows.\n", rows_in_charge);
// #endif

	int* my_rows = (int*)malloc(sizeof(int) * mat_size * rows_in_charge);
	int* i_to_k = (int*)malloc(sizeof(int) * mat_size); // the horizontal (column)
	int* k_to_j = (int*)malloc(sizeof(int) * mat_size); // the vertical (row)

	// divide the matrix for each process
	// TODO send rows to each process using scatter, sendbuf:*result, recvbuf:*my_rows
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
	int* distribute_row_size = (int*)malloc(sizeof(int) * total_rank);
	int* distribute_row_offset = (int*)malloc(sizeof(int) * total_rank);
	int* distribute_ele_size = (int*)malloc(sizeof(int) * total_rank);
	int* distribute_ele_offset = (int*)malloc(sizeof(int) * total_rank);
	int mainly_have_rows = mat_size % total_rank == 0 ? mat_size / total_rank : mat_size / total_rank + 1;
	for (int i = 0; i < total_rank; i++)
		distribute_row_size[i] = mainly_have_rows;
	if (mat_size % mainly_have_rows != 0)
		distribute_row_size[total_rank - 1] = mat_size % mainly_have_rows;
	int accu = 0;
	for (int i = 0; i < total_rank; i++) {
		distribute_row_offset[i] = accu;
		accu += distribute_row_size[i];
	}
	for (int i = 0; i < total_rank; i++) {
		distribute_ele_size[i] = mat_size * distribute_row_size[i];
		distribute_ele_offset[i] = mat_size * distribute_row_offset[i];
	}

// #ifdef DEBUG_MSG
// 	shout_name();
// 	printf("distribute_ele_size: \n");
// 	show_arr(distribute_ele_size, total_rank);
// 	printf("distribute_ele_offset: \n");
// 	show_arr(distribute_ele_offset, total_rank);
// #endif

	int recvcount = mat_size * rows_in_charge;
	MPI_Scatterv(
		result, 
		distribute_ele_size, 
		distribute_ele_offset, 
		MPI_INT, 
		my_rows, 
		recvcount, 
		MPI_INT, 
		0, 
		MPI_COMM_WORLD);

#ifdef DEBUG_MSG
	shout_name();
	printf("my_rows: \n");
	show_dd_matrix(my_rows, rows_in_charge, mat_size);
#endif

	// set -1 edge (disconnected) to INT_MAX / 2
	for (int i = row_start; i < row_end; i++)
		for (int j = 0; j < mat_size; j++) {
			int cur_index = i * mat_size + j;
			if (my_rows[cur_index] < 0)
				my_rows[cur_index] = INT_MAX / 2;
		}

	// for k:=0 to n 
	//		for i:=rowStart to rowEnd
	//			for j:=0 to n
	//				my_rows[i,j]=min(my_rows[i,j],my_rows[i,k]+my_rows[k,j]);
	//		bcast result to other processes. (bcast or allgather?), at the end of each round, process 0 has all the resultsss
	for (int k = 0; k < mat_size; k++) {
		// TODO broadcast k-th row to other process if I am the owner
		// MPI_Bcast(
		//     void* data,
		//     int count,
		//     MPI_Datatype datatype,
		//     int root,
		//     MPI_Comm communicator)

		int owner_of_k_row = k / mainly_have_rows;
		if (my_rank == owner_of_k_row)
			memcpy(k_to_j, my_rows + mat_size * (k % mainly_have_rows), sizeof(int) * mat_size);
		MPI_Bcast((void*)k_to_j, mat_size, MPI_INT, owner_of_k_row, MPI_COMM_WORLD);

		// TODO broadcase k-th column to other process (put col into i_to_k first
		// TODO and then send/recv to complete i_to_k)
		// MPI_Gather(
		// 	void* send_data,
		// 	int send_count,
		// 	MPI_Datatype send_datatype,
		// 	void* recv_data,
		// 	int recv_count,
		// 	MPI_Datatype recv_datatype,
		// 	int root,
		// 	MPI_Comm communicator)
		// MPI_Allgather(
		//     void* send_data,
		//     int send_count,
		//     MPI_Datatype send_datatype,
		//     void* recv_data,
		//     int recv_count,
		//     MPI_Datatype recv_datatype,
		//     MPI_Comm communicator)
		// int MPI_Allgatherv(
		// 	const void *sendbuf, 
		// 	int sendcount, 
		// 	MPI_Datatype sendtype,
		// 	void *recvbuf, 
		// 	const int *recvcounts, 
		// 	const int *displs,
		// 	MPI_Datatype recvtype, 
		// 	MPI_Comm comm)

		for (int i = row_start; i < row_end; i++)
			i_to_k[my_rank * mainly_have_rows + i] = my_rows[i * mat_size + k];

		MPI_Allgatherv(
			i_to_k + my_rank * mainly_have_rows, 
			rows_in_charge,
			MPI_INT,
			i_to_k,
			distribute_row_size,
			distribute_row_offset,
			MPI_INT,
			MPI_COMM_WORLD);

		if (my_rank == 1)
			MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG_MSG
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
#endif

		if (my_rank != 1)
			MPI_Barrier(MPI_COMM_WORLD);

		for (int i = row_start; i < row_end; i++) {
			for (int j = 0; j < mat_size; j++) {
				int use_k = i_to_k[mainly_have_rows * my_rank + i] + k_to_j[j];
				int cur_index = i * mat_size + j;
				my_rows[cur_index] = min(my_rows[cur_index], use_k);
			}
		}

#ifdef DEBUG_MSG
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gatherv(
			my_rows,
			mat_size * rows_in_charge,
			MPI_INT,
			result,
			distribute_ele_size,
			distribute_ele_offset,
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

	// TODO collect result to process 0
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
		mat_size * rows_in_charge,
		MPI_INT,
		result,
		distribute_ele_size,
		distribute_ele_offset,
		MPI_INT,
		0,
		MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	if (my_rank == 0) {
		//stop the timer
		gettimeofday(&tv2, NULL);
		time_used_parallel = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
		printf("Time used (parallel  ): %8ld usecs\n", time_used_parallel);
		printf("Speed up (sequential / parallel): %.3lf\n", time_used_sequential / (double)time_used_parallel);

#ifdef DEBUG_MSG
		printf("Correct Answer: \n");
		show_dd_matrix(ref, mat_size, mat_size);
		printf("\n");
#endif
#ifdef DEBUG_MSG
		printf("FINAL Result: \n");
		show_dd_matrix(result, mat_size, mat_size);
		printf("\n");
#endif
		//compare your result with reference result
		if(CmpArray(result, ref, mat_size * mat_size))
			printf("Your result is correct.\n");
		else
			printf("Your result is wrong.\n");
	}

    // Finalize the MPI environment.
    MPI_Finalize();
}
