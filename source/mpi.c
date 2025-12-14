#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "types.h"

uint32_t point_amount = 0u;
uint32_t centroid_amount = 0u;
uint32_t iteration_counter = 0u;
uint32_t iteration_limit = 0u;
float64_t sum_squared_errors = 0.0f;
float64_t epsilon = 0.0f;

float64_t *points_global = NULL;
uint32_t *assignments_global = NULL;

uint32_t local_point_count = 0u;
float64_t *points_local = NULL;
uint32_t *assignments_local = NULL;

float64_t *centroids = NULL;

float64_t *sum_centroid_points_local = NULL;
uint32_t *amount_centroid_points_local = NULL;
float64_t *sum_centroid_points_global = NULL;
uint32_t *amount_centroid_points_global = NULL;

int *counts = NULL;
int *displs = NULL;

int mpi_rank = 0;
int mpi_size = 1;

double comm_time_total = 0.0;

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

    double *A = malloc((unsigned long) R * sizeof(double));
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
            fprintf(stderr, "%s: linha %d sem valor\n", path, r+1);
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

static void write_assign_csv(const char *path, const uint32_t *assign, uint32_t N)
{
    if (!path) return;

    FILE *f = fopen(path, "w");
    if (!f)
    {
        perror(path);
        return;
    }
    for (uint32_t i = 0; i < N; i++)
    {
        fprintf(f, "%u\n", assign[i]);
    }
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, uint32_t K)
{
    if (!path) return;

    FILE *f = fopen(path, "w");
    if (!f)
    {
        perror(path);
        return;
    }
    for (uint32_t c = 0; c < K; c++)
    {
        fprintf(f, "%.6f\n", C[c]);
    }
    fclose(f);
}

static void distribute_points(void)
{
    counts = malloc((size_t) mpi_size * sizeof(int));
    displs = malloc((size_t) mpi_size * sizeof(int));
    
    if (!counts || !displs)
    {
        perror("malloc counts/displs");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int base = (int) point_amount / mpi_size;
    int remainder = (int) point_amount % mpi_size;
    int offset = 0;
    
    for (int p = 0; p < mpi_size; p++)
    {
        counts[p] = base + (p < remainder ? 1 : 0);
        displs[p] = offset;
        offset += counts[p];
    }
    
    local_point_count = (uint32_t) counts[mpi_rank];

    points_local = malloc((size_t) local_point_count * sizeof(float64_t));
    assignments_local = malloc((size_t) local_point_count * sizeof(uint32_t));

    if (!points_local || !assignments_local)
    {
        perror("malloc local");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double comm_start = MPI_Wtime();
    MPI_Scatterv(points_global, counts, displs, MPI_DOUBLE,
                 points_local, (int) local_point_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    comm_time_total += MPI_Wtime() - comm_start;
}

static void assignment_step_local(float64_t *sse_local)
{
    *sse_local = 0.0;

    for (uint32_t i = 0; i < local_point_count; i++)
    {
        uint32_t best_centroid = 0u;
        float64_t min_dist_sq = 1e300;

        for (uint32_t c = 0; c < centroid_amount; c++)
        {
            float64_t error = points_local[i] - centroids[c];
            float64_t dist_sq = error * error;

            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                best_centroid = c;
            }
        }

        assignments_local[i] = best_centroid;
        *sse_local += min_dist_sq;
    }
}

static void update_step_local(void)
{
    memset(sum_centroid_points_local, 0, (size_t) centroid_amount * sizeof(float64_t));
    memset(amount_centroid_points_local, 0, (size_t) centroid_amount * sizeof(uint32_t));

    for (uint32_t i = 0; i < local_point_count; i++)
    {
        uint32_t c = assignments_local[i];
        sum_centroid_points_local[c] += points_local[i];
        amount_centroid_points_local[c]++;
    }
}

static void kmeans_mpi(void)
{
    float64_t sse_holder = 1e300;
    float64_t sse_local = 0.0;

    sum_centroid_points_local = calloc((size_t) centroid_amount, sizeof(float64_t));
    amount_centroid_points_local = calloc((size_t) centroid_amount, sizeof(uint32_t));
    sum_centroid_points_global = calloc((size_t) centroid_amount, sizeof(float64_t));
    amount_centroid_points_global = calloc((size_t) centroid_amount, sizeof(uint32_t));

    if (!sum_centroid_points_local || !amount_centroid_points_local ||
        !sum_centroid_points_global || !amount_centroid_points_global)
    {
        perror("calloc");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double comm_start = MPI_Wtime();
    MPI_Bcast(centroids, (int) centroid_amount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_time_total += MPI_Wtime() - comm_start;

    for (iteration_counter = 0; iteration_counter < iteration_limit; iteration_counter++)
    {
        assignment_step_local(&sse_local);
        update_step_local();

        comm_start = MPI_Wtime();
        MPI_Reduce(&sse_local, &sum_squared_errors, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(sum_centroid_points_local, sum_centroid_points_global,
                     (int) centroid_amount, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(amount_centroid_points_local, amount_centroid_points_global,
                     (int) centroid_amount, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        comm_time_total += MPI_Wtime() - comm_start;

        for (uint32_t c = 0; c < centroid_amount; c++)
        {
            if (amount_centroid_points_global[c] > 0)
            {
                centroids[c] = sum_centroid_points_global[c] / amount_centroid_points_global[c];
            }
        }

        int converged = 0;
        if (mpi_rank == 0)
        {
            float64_t relative_change = fabs(sum_squared_errors - sse_holder) /
                                       (sse_holder > 0.0 ? sse_holder : 1.0);
            if (relative_change < epsilon)
            {
                converged = 1;
            }
            sse_holder = sum_squared_errors;
        }

        comm_start = MPI_Wtime();
        MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
        comm_time_total += MPI_Wtime() - comm_start;

        if (converged)
        {
            iteration_counter++;
            break;
        }
    }

    free(sum_centroid_points_local);
    free(amount_centroid_points_local);
    free(sum_centroid_points_global);
    free(amount_centroid_points_global);
}

static void gather_assignments(void)
{
    double comm_start = MPI_Wtime();
    MPI_Gatherv(assignments_local, (int) local_point_count, MPI_UNSIGNED,
                assignments_global, counts, displs, MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    comm_time_total += MPI_Wtime() - comm_start;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 3)
    {
        if (mpi_rank == 0)
        {
            printf("Uso: mpirun -np P %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
            printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *path_points_arg = argv[1];
    const char *path_centroids_arg = argv[2];
    iteration_limit = (argc > 3) ? (uint32_t) atoi(argv[3]) : 50u;
    epsilon = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *path_assignment_arg = (argc > 5) ? argv[5] : NULL;
    const char *path_output_centroid_arg = (argc > 6) ? argv[6] : NULL;

    if (iteration_limit <= 0 || epsilon <= 0.0)
    {
        if (mpi_rank == 0)
            fprintf(stderr, "max_iter>0 e eps>0\n");
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0)
    {
        points_global = read_csv_1col(path_points_arg, &point_amount);
        centroids = read_csv_1col(path_centroids_arg, &centroid_amount);
        assignments_global = malloc((size_t) point_amount * sizeof(uint32_t));

        if (!assignments_global)
        {
            perror("malloc");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&point_amount, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&centroid_amount, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0)
    {
        centroids = malloc((size_t) centroid_amount * sizeof(float64_t));
        if (!centroids)
        {
            perror("malloc");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    distribute_points();

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    kmeans_mpi();

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_ms = (end_time - start_time) * 1000.0;

    gather_assignments();

    if (mpi_rank == 0)
    {
        printf("K-means 1D (MPI)\n");
        printf("N=%u K=%u max_iter=%u eps=%g threads=%d\n",
               point_amount, centroid_amount, iteration_limit, epsilon, mpi_size);
        printf("Iterações: %u | SSE final: %.10f\n", iteration_counter, sum_squared_errors);
        printf("Tempo total: %.6f ms | Tempo comunicação: %.6f ms (%.2f%%)\n",
               elapsed_ms, comm_time_total * 1000.0,
               (comm_time_total * 1000.0 / elapsed_ms) * 100.0);

        write_assign_csv(path_assignment_arg, assignments_global, point_amount);
        write_centroids_csv(path_output_centroid_arg, centroids, centroid_amount);

        free(points_global);
        free(assignments_global);
    }

    free(points_local);
    free(assignments_local);
    free(centroids);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}