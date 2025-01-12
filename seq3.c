#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ELEMENTS 100000 // 1 lakh

// Swap function to exchange two elements
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// 3-way partition QuickSort implementation
void three_way_quicksort(int* arr, int low, int high) {
    if (low >= high) return;

    // Choose pivot
    int pivot = arr[low];
    int left = low;
    int right = high;
    int i = low + 1;

    while (i <= right) {
        if (arr[i] < pivot) {
            swap(&arr[i], &arr[left]);
            left++;
            i++;
        } else if (arr[i] > pivot) {
            swap(&arr[i], &arr[right]);
            right--;
        } else {
            i++;
        }
    }

    // Recursively sort partitions
    three_way_quicksort(arr, low, left - 1);   // Elements less than pivot
    three_way_quicksort(arr, right + 1, high); // Elements greater than pivot
}

// Function to verify if the array is sorted
int verify_sorted(int* arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) {
            return 0;
        }
    }
    return 1;
}

int main() {
    int n;
    int* arr;

    // Open the input file
    FILE* input_file = fopen("input.txt", "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error opening input file\n");
        return 1;
    }

    // Read the number of elements
    fscanf(input_file, "%d", &n);

    // Allocate memory for the array
    arr = (int*)malloc(n * sizeof(int));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(input_file);
        return 1;
    }

    // Read the array elements
    for (int i = 0; i < n; i++) {
        fscanf(input_file, "%d", &arr[i]);
    }

    // Close the input file
    fclose(input_file);

    // Record the start time
    clock_t start = clock();

    // Perform 3-way QuickSort
    three_way_quicksort(arr, 0, n - 1);

    // Record the end time
    clock_t end = clock();

    // Verify if the array is sorted
    if (verify_sorted(arr, n)) {
        printf("Array is sorted\n");
    } else {
        printf("Array is NOT sorted\n");
    }

    // Calculate the time taken in milliseconds
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC * 1000;  // Convert to milliseconds
    printf("Time taken to sort %d elements: %.3f milliseconds\n", n, time_taken);

    // Write the sorted array to output file
    FILE* output_file = fopen("output.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error opening output file\n");
        free(arr);
        return 1;
    }

    // Write the number of elements
    fprintf(output_file, "%d\n", n);

    // Write the sorted array elements
    for (int i = 0; i < n; i++) {
        fprintf(output_file, "%d ", arr[i]);
    }
    fprintf(output_file, "\n");

    // Close the output file
    fclose(output_file);

    // Free allocated memory
    free(arr);

    return 0;
}

