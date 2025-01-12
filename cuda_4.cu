#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 6
#define MAX_SEGMENTS 256
#define INSERTION_SORT_THRESHOLD 32

__device__ inline void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ void insertion_sort(int* arr, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__device__ int partition(int* arr, int low, int high, int* lt, int* gt) {
    if (high - low < INSERTION_SORT_THRESHOLD) {
        insertion_sort(arr, low, high);
        *lt = low;
        *gt = high;
        return low;
    }

    // Choose pivot using median-of-three
    int mid = low + (high - low) / 2;
    if (arr[mid] < arr[low]) swap(arr[low], arr[mid]);
    if (arr[high] < arr[low]) swap(arr[low], arr[high]);
    if (arr[mid] < arr[high]) swap(arr[mid], arr[high]);
    
    int pivot = arr[high];
    int i = low;
    *lt = low;
    *gt = high;
    
    while (i <= *gt) {
        if (arr[i] < pivot) {
            swap(arr[*lt], arr[i]);
            (*lt)++;
            i++;
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[*gt]);
            (*gt)--;
        } else {
            i++;
        }
    }
    
    return *lt;
}

__device__ void three_way_quicksort_kernel(int* arr, int low, int high) {
    while (low < high) {
        if (high - low < INSERTION_SORT_THRESHOLD) {
            insertion_sort(arr, low, high);
            return;
        }

        int lt, gt;
        partition(arr, low, high, &lt, &gt);
        
        // Tail recursion optimization
        if (lt - low < high - gt) {
            three_way_quicksort_kernel(arr, low, lt - 1);
            low = gt + 1;
        } else {
            three_way_quicksort_kernel(arr, gt + 1, high);
            high = lt - 1;
        }
    }
}

__global__ void parallel_three_way_quicksort(int* arr, int n, int segment_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int segment_start = tid * segment_size;
    
    if (segment_start < n) {
        int segment_end = min(segment_start + segment_size - 1, n - 1);
        three_way_quicksort_kernel(arr, segment_start, segment_end);
    }
}

__device__ void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i <= mid) temp[k++] = arr[i++];
    while (j <= right) temp[k++] = arr[j++];
    
    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

__global__ void parallel_merge(int* arr, int* temp, int n, int segment_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int left = tid * (segment_size * 2);
    
    if (left < n) {
        int mid = min(left + segment_size - 1, n - 1);
        int right = min(left + (segment_size * 2) - 1, n - 1);
        if (mid < right) {
            merge(arr, temp, left, mid, right);
        }
    }
}

void check_cuda_error(cudaError_t error, const char* function_name) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", function_name, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n;
    int* h_arr;
    int* d_arr;
    int* d_temp;
    cudaError_t cuda_status;

    // Read input from file
    FILE* input_file = fopen("input.txt", "r");
    if (!input_file) {
        fprintf(stderr, "Error opening input file\n");
        return 1;
    }

    fscanf(input_file, "%d", &n);
    h_arr = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        fscanf(input_file, "%d", &h_arr[i]);
    }
    fclose(input_file);

    printf("Read %d elements from input file\n", n);

    // Allocate device memory with aligned access
    cuda_status = cudaMalloc((void**)&d_arr, n * sizeof(int));
    check_cuda_error(cuda_status, "cudaMalloc d_arr");
    cuda_status = cudaMalloc((void**)&d_temp, n * sizeof(int));
    check_cuda_error(cuda_status, "cudaMalloc d_temp");

    // Copy input data to device
    cuda_status = cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "cudaMemcpy H2D");

    // Create CUDA events
    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float milliseconds1, milliseconds= 0;
    float mtime=0;
    // Calculate optimal segment size and grid dimensions
    int segment_size = (n + MAX_SEGMENTS - 1) / MAX_SEGMENTS;
    segment_size = max(segment_size, INSERTION_SORT_THRESHOLD);
    
    int num_segments = (n + segment_size - 1) / segment_size;
    int num_blocks = (num_segments + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch sorting kernel
    cudaEventRecord(start1, 0);
    parallel_three_way_quicksort<<<num_blocks, BLOCK_SIZE>>>(d_arr, n, segment_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop1, 0);
	
    // Merge sorted segments
    for (int curr_size = segment_size; curr_size < n; curr_size *= 2) {
        int merge_blocks = (n + curr_size * 2 - 1) / (curr_size * 2);
        merge_blocks = (merge_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaEventRecord(start2, 0);
        parallel_merge<<<merge_blocks, BLOCK_SIZE>>>(d_arr, d_temp, n, curr_size);
        cudaEventRecord(stop2, 0);
        cudaEventElapsedTime(&milliseconds1, start2, stop2);
        cudaDeviceSynchronize();
        mtime+=milliseconds1;
    }

    // Record time and check for errors
    
    
    
    cudaEventElapsedTime(&milliseconds, start1, stop1);
	
    cuda_status = cudaGetLastError();
    check_cuda_error(cuda_status, "kernel execution");

    // Copy result back to host
    cuda_status = cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    check_cuda_error(cuda_status, "cudaMemcpy D2H");

    // Verify and write output
    bool is_sorted = true;
    for (int i = 1; i < n && is_sorted; i++) {
        if (h_arr[i] < h_arr[i-1]) is_sorted = false;
    }
    printf("Array is %s\n", is_sorted ? "sorted" : "not sorted");

    FILE* output_file = fopen("output.txt", "w");
    if (output_file) {
        fprintf(output_file, "%d\n", n);
        for (int i = 0; i < n; i++) {
            fprintf(output_file, "%d ", h_arr[i]);
        }
        fclose(output_file);
        printf("Sorted %d elements and wrote to output file\n", n);
        printf("Sorting time: %f milliseconds\n", milliseconds+mtime);
    }

    // Cleanup
    cudaFree(d_arr);
    cudaFree(d_temp);
    free(h_arr);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    return 0;
}
