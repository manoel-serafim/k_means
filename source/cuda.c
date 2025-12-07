%%writefile kmeans_1d.cu

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

typedef double float64_t;
typedef unsigned int uint32_t;

uint32_t point_amount = 0u;
uint32_t centroid_amount = 0u;
uint32_t iteration_counter = 0u;
uint32_t iteration_limit = 0u;
float64_t sum_squared_errors = 0.0f;
float64_t epsilon = 0.0f;

const float64_t *points = NULL;
float64_t *centroids = NULL;
uint32_t *assignments = NULL;

float64_t *d_points = NULL;
float64_t *d_centroids = NULL;
uint32_t *d_assignments = NULL;
float64_t *d_sse_array = NULL;
float64_t *d_sum_centroid_points = NULL;
uint32_t *d_amount_centroid_points = NULL;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static uint32_t count_rows(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
    {
        perror(path);
        exit(1);
    }
    uint32_t rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f))
    {
        for (char *p = line; *p; p++)
        {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            {
                rows++;
                break;
            }
        }
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, uint32_t *n_out)
{
    uint32_t R = count_rows(path);
    if (R <= 0)
    {
        fprintf(stderr, "%s: arquivo vazio\n", path);
        exit(1);
    }

    double *A = (double*) malloc((unsigned long) R * sizeof(double));

    if (!A)
    {
        perror("malloc");
        exit(1);
    }

    FILE *f = fopen(path, "r");
    if (!f)
    {
        perror(path);
        free(A);
        exit(1);
    }

    char line[8192];
    uint32_t r = 0;
    while (fgets(line, sizeof(line), f) && r < R)
    {
        char *tok = strtok(line, ",; \t\n\r");
        if (!tok)
        {
            fprintf(stderr, "%s: linha %d sem valor\n", path, (int) (r+1));
            free(A);
            fclose(f);
            exit(1);
        }
        A[r++] = atof(tok);
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path,
                             const uint32_t *assign,
                             uint32_t N)
{
    if (!path)
        return;

    FILE *f = fopen(path, "w");
    if (!f)
    {
        perror(path);
        return;
    }
    for (uint32_t i = 0; i < N; i++)
        fprintf(f, "%u\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path,
                                const double *C,
                                uint32_t K)
{
    if (!path)
        return;

    FILE *f = fopen(path, "w");
    if (!f)
    {
        perror(path);
        return;
    }
    for (uint32_t c = 0; c < K; c++)
        fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

__global__ void assignment_kernel(
    const float64_t *points,
    const float64_t *centroids,
    uint32_t *assignments,
    float64_t *sse_array,
    uint32_t point_amount,
    uint32_t centroid_amount)
{
    uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= point_amount)
        return;

    uint32_t best_centroid = 0;
    float64_t error = points[point_index] - centroids[0];
    float64_t min_squared_error = error * error;

    for (uint32_t centroid_index = 1; centroid_index < centroid_amount; ++centroid_index)
    {
        error = points[point_index] - centroids[centroid_index];
        float64_t squared_error = error * error;
        if (squared_error < min_squared_error)
        {
            min_squared_error = squared_error;
            best_centroid = centroid_index;
        }
    }

    assignments[point_index] = best_centroid;
    sse_array[point_index] = min_squared_error;
}

__global__ void update_kernel(
    const float64_t *points,
    const uint32_t *assignments,
    float64_t *sum_centroid_points,
    uint32_t *amount_centroid_points,
    uint32_t point_amount)
{
    uint32_t point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= point_amount)
        return;

    uint32_t centroid_index = assignments[point_index];
    atomicAdd(&sum_centroid_points[centroid_index], points[point_index]);
    atomicAdd(&amount_centroid_points[centroid_index], 1u);
}

__global__ void compute_centroids_kernel(
    float64_t *centroids,
    const float64_t *sum_centroid_points,
    const uint32_t *amount_centroid_points,
    const float64_t *points,
    uint32_t centroid_amount)
{
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= centroid_amount)
        return;

    if (amount_centroid_points[c] > 0)
        centroids[c] = sum_centroid_points[c] / (float64_t)amount_centroid_points[c];
    else
        centroids[c] = points[0];
}

static void assignment_step_1d(int block_size)
{
    int grid_size = (point_amount + block_size - 1) / block_size;

    assignment_kernel<<<grid_size, block_size>>>(
        d_points, d_centroids, d_assignments, d_sse_array,
        point_amount, centroid_amount);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float64_t *h_sse_array = (float64_t*)malloc(point_amount * sizeof(float64_t));
    CUDA_CHECK(cudaMemcpy(h_sse_array, d_sse_array,
                          point_amount * sizeof(float64_t), cudaMemcpyDeviceToHost));

    float64_t local_sse = 0.0;
    for (uint32_t i = 0; i < point_amount; ++i)
        local_sse += h_sse_array[i];

    sum_squared_errors = local_sse;
    free(h_sse_array);
}

static void update_step_1d_gpu(int block_size)
{
    int grid_size = (point_amount + block_size - 1) / block_size;

    CUDA_CHECK(cudaMemset(d_sum_centroid_points, 0, centroid_amount * sizeof(float64_t)));
    CUDA_CHECK(cudaMemset(d_amount_centroid_points, 0, centroid_amount * sizeof(uint32_t)));

    update_kernel<<<grid_size, block_size>>>(
        d_points, d_assignments, d_sum_centroid_points,
        d_amount_centroid_points, point_amount);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int centroid_grid = (centroid_amount + block_size - 1) / block_size;
    compute_centroids_kernel<<<centroid_grid, block_size>>>(
        d_centroids, d_sum_centroid_points, d_amount_centroid_points,
        d_points, centroid_amount);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(centroids, d_centroids,
                          centroid_amount * sizeof(float64_t), cudaMemcpyDeviceToHost));
}

static void kmeans_1d(int block_size)
{
    float64_t sse_holder = 1e300;
    float64_t relative_change = 0.0f;

    for (iteration_counter = 0; iteration_counter < iteration_limit; ++iteration_counter)
    {
        assignment_step_1d(block_size);

        relative_change = fabs(sum_squared_errors - sse_holder) /
                          (sse_holder > 0.0 ? sse_holder : 1.0);

        if (relative_change < epsilon)
        {
            ++iteration_counter;
            break;
        }
        else
        {
            update_step_1d_gpu(block_size);
            sse_holder = sum_squared_errors;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [block_size=256]\n", argv[0]);
        return 1;
    }

    const char *const path_points = argv[1];
    const char *const path_centroids = argv[2];
    iteration_limit = (argc > 3) ? (uint32_t) atoi(argv[3]) : 50u;
    epsilon = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *path_assignment = (argc > 5) ? argv[5] : NULL;
    const char *path_output_centroid = (argc > 6) ? argv[6] : NULL;
    int block_size = (argc > 7) ? atoi(argv[7]) : 256;

    points = read_csv_1col(path_points, &point_amount);
    centroids = read_csv_1col(path_centroids, &centroid_amount);
    assignments = (uint32_t*) malloc((size_t) point_amount * sizeof(uint32_t));

    CUDA_CHECK(cudaMalloc(&d_points, point_amount * sizeof(float64_t)));
    CUDA_CHECK(cudaMalloc(&d_centroids, centroid_amount * sizeof(float64_t)));
    CUDA_CHECK(cudaMalloc(&d_assignments, point_amount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sse_array, point_amount * sizeof(float64_t)));
    CUDA_CHECK(cudaMalloc(&d_sum_centroid_points, centroid_amount * sizeof(float64_t)));
    CUDA_CHECK(cudaMalloc(&d_amount_centroid_points, centroid_amount * sizeof(uint32_t)));

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));

    CUDA_CHECK(cudaEventRecord(start_total));

    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_points, points, point_amount * sizeof(float64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids, centroid_amount * sizeof(float64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_h2d));
    CUDA_CHECK(cudaEventSynchronize(stop_h2d));

    CUDA_CHECK(cudaEventRecord(start_kernel));
    kmeans_1d(block_size);
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(assignments, d_assignments, point_amount * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(centroids, d_centroids, centroid_amount * sizeof(float64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_d2h));
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    float time_h2d, time_kernel, time_d2h, time_total;
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h));
    CUDA_CHECK(cudaEventElapsedTime(&time_total, start_total, stop_total));

    printf("K-means 1D (CUDA - GPU Atomics)\n");
    printf("N=%u K=%u max_iter=%u eps=%g block_size=%d\n",
           point_amount, centroid_amount, iteration_limit, epsilon, block_size);
    printf("Iterações: %u | SSE final: %.10f\n",
           iteration_counter, sum_squared_errors);
    printf("Tempo H2D: %.3f ms | Kernel: %.3f ms | D2H: %.3f ms | Total: %.3f ms\n",
           time_h2d, time_kernel, time_d2h, time_total);

    write_assign_csv(path_assignment, assignments, point_amount);
    write_centroids_csv(path_output_centroid, centroids, centroid_amount);

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_sse_array));
    CUDA_CHECK(cudaFree(d_sum_centroid_points));
    CUDA_CHECK(cudaFree(d_amount_centroid_points));
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));

    free(assignments);
    free(centroids);
    free((void *) points);
    return 0;
}
