// C program to implement the Quick Sort
// Algorithm using MPI
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Function to swap two numbers
void swap(int* arr, int i, int j)
{
    int t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

// Function that performs the 3-way Quick Sort
void threequicksort(int* arr, int low, int high)
{
    if (high <= low) return;

    int pivot = arr[low];
    int lt = low;    // Pointer for the less than section
    int gt = high;   // Pointer for the greater than section
    int i = low + 1; // Current index

    while (i <= gt) {
        if (arr[i] < pivot) {
            swap(arr, lt++, i++);
        } else if (arr[i] > pivot) {
            swap(arr, i, gt--);
        } else {
            i++;
        }
    }

    // Recursively sort the three partitions
    threequicksort(arr, low, lt - 1);
    threequicksort(arr, gt + 1, high);
}

// Function that merges two arrays
int* merge(int* arr1, int n1, int* arr2, int n2)
{
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0, j = 0, k;

    for (k = 0; k < n1 + n2; k++) {
        if (i >= n1) {
            result[k] = arr2[j++];
        } else if (j >= n2) {
            result[k] = arr1[i++];
        } else if (arr1[i] < arr2[j]) {
            result[k] = arr1[i++];
        } else {
            result[k] = arr2[j++];
        }
    }
    return result;
}

// Driver Code
int main(int argc,char* argv[])
{
    int number_of_elements;
    int* data = NULL;
    int chunk_size, own_chunk_size;
    int* chunk;
    FILE* file = NULL;
    double time_taken = 0.0;
    MPI_Status status;


    int number_of_process, rank_of_process;
    int rc = MPI_Init(&argc, &argv);

    if (rc != MPI_SUCCESS) {
        printf("Error in creating MPI program.\nTerminating......\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process);

    if (rank_of_process == 0) {
        // Opening the file
        file = fopen("input.txt", "r");

        // Check if file opened successfully
        if (file == NULL) {
            printf("Error in opening file\n");
            exit(-1);
        }

        // Reading number of Elements in file
        fscanf(file, "%d", &number_of_elements);
        printf("Number of Elements in the file is %d \n", number_of_elements);

        // Computing chunk size
        chunk_size = (number_of_elements % number_of_process == 0)
                         ? (number_of_elements / number_of_process)
                         : (number_of_elements / number_of_process - 1);

        data = (int*)malloc(number_of_process * chunk_size * sizeof(int));

        // Reading the elements from the file
        for (int i = 0; i < number_of_elements; i++) {
            fscanf(file, "%d", &data[i]);
        }

        // Padding data with zero
        for (int i = number_of_elements; i < number_of_process * chunk_size; i++) {
            data[i] = 0;
        }

        fclose(file);
        file = NULL;
    }

    // Synchronize processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast the size to all processes
    MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Computing chunk size
    chunk_size = (number_of_elements % number_of_process == 0)
                     ? (number_of_elements / number_of_process)
                     : number_of_elements / (number_of_process - 1);

    // Allocate memory for the chunk
    chunk = (int*)malloc(chunk_size * sizeof(int));

    // Scatter the chunk data to all processes
    MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(data);
    data = NULL;

    // Compute size of own chunk and sort it
    own_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 1))
                         ? chunk_size
                         : (number_of_elements - chunk_size * rank_of_process);

    // Start timer
    time_taken -= MPI_Wtime();
    threequicksort(chunk, 0, own_chunk_size);
    
	
    // Merging sorted chunks
    for (int step = 1; step < number_of_process; step = 2 * step) {
        if (rank_of_process % (2 * step) != 0) {
            MPI_Send(chunk, own_chunk_size, MPI_INT, rank_of_process - step, 0, MPI_COMM_WORLD);
            break;
        }

        if (rank_of_process + step < number_of_process) {
            int received_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 2 * step))
                                           ? (chunk_size * step)
                                           : (number_of_elements - chunk_size * (rank_of_process + step));
            int* chunk_received = (int*)malloc(received_chunk_size * sizeof(int));
            MPI_Recv(chunk_received, received_chunk_size, MPI_INT, rank_of_process + step, 0, MPI_COMM_WORLD, &status);

            data = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);

            free(chunk);
            free(chunk_received);
            chunk = data;
            own_chunk_size += received_chunk_size;
        }
    }
    // Stop the timer
    time_taken += MPI_Wtime();
	
    

    // Output results
    if (rank_of_process == 0) {
        file = fopen("output.txt", "w");

        if (file == NULL) {
            printf("Error in opening output file... \n");
            exit(-1);
        }

        fprintf(file, "Total number of Elements in the array: %d\n", own_chunk_size);
        for (int i = 0; i < own_chunk_size; i++) {
            fprintf(file, "%d ", chunk[i]);
        }

        fclose(file);
        printf("\nSorted array written to output.txt\n");
/*
        printf("Total number of Elements given as input: %d\n", number_of_elements);
        printf("Sorted array is: \n");
        for (int i = 0; i < own_chunk_size; i++) {
            printf("%d ", chunk[i]);
        }
        */
        printf("\nThree way Quicksort %d ints on %d procs: %f millisecs\n", number_of_elements, number_of_process, time_taken * 1000);
    }

    MPI_Finalize();
    return 0;
}
