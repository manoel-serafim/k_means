/* openmp.c
   K-means 1D com paralelização OpenMP (Opção A - acumuladores por thread)
   - Lê X (N linhas, 1 coluna) e C_init (K linhas, 1 coluna) de CSVs sem cabeçalho.
   - Itera assignment + update até max_iter ou variação relativa do SSE < eps.
   - Salva (opcional) assign (N linhas) e centróides finais (K linhas).
   - Paralelização: assignment com reduction, update com acumuladores privados.

   Compilar: clang -O3 -fopenmp openmp.c -o openmp -lm
   Uso:      ./openmp dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [num_threads=4]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "types.h"

uint32_t point_amount = 0u;
uint32_t centroid_amount = 0u;
uint32_t iteration_counter = 0u;
uint32_t iteration_limit = 0u;
float64_t sum_squared_errors = 0.0f;
float64_t epsilon = 0.0f;

const float64_t *points = NULL;
float64_t *centroids = NULL;
float64_t *sum_centroid_points = NULL;
uint32_t *amount_centroid_points = NULL;
uint32_t *assignments = NULL;

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
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

static double *read_csv_1col(const char *path,
                             uint32_t *n_out)
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
    {
        return;
    }

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

static void write_centroids_csv(const char *path,
                                const double *C,
                                uint32_t K)
{
    if (!path)
    {
        return;
    }

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

static void assignment_step_1d(void)
{
    float64_t local_sse = 0.0;

    #pragma omp parallel for reduction(+:local_sse) schedule(static)
    for (uint32_t point_index = 0; point_index < point_amount; ++point_index)
    {
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
        local_sse += min_squared_error;
    }

    sum_squared_errors = local_sse;
}

/* Update step paralelo com acumuladores por thread (Opção A) */
static void update_step_1d(void)
{
    uint32_t num_threads = (uint32_t) omp_get_max_threads();

    float64_t *thread_sums = calloc((size_t) (num_threads * centroid_amount), sizeof(float64_t));
    uint32_t *thread_counts = calloc((size_t) (num_threads * centroid_amount), sizeof(uint32_t));

    if (!thread_sums || !thread_counts)
    {
        perror("calloc thread arrays");
        exit(1);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float64_t *my_sums = &thread_sums[(uint32_t) tid * centroid_amount];
        uint32_t *my_counts = &thread_counts[(uint32_t) tid * centroid_amount];

        #pragma omp for schedule(static)
        for (uint32_t point_index = 0; point_index < point_amount; ++point_index)
        {
            uint32_t centroid_index = assignments[point_index];
            my_counts[centroid_index]++;
            my_sums[centroid_index] += points[point_index];
        }
    }

    memset(sum_centroid_points, 0, centroid_amount * sizeof(float64_t));
    memset(amount_centroid_points, 0, centroid_amount * sizeof(uint32_t));

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        for (uint32_t c = 0; c < centroid_amount; ++c)
        {
            sum_centroid_points[c] += thread_sums[t * centroid_amount + c];
            amount_centroid_points[c] += thread_counts[t * centroid_amount + c];
        }
    }

    #pragma omp parallel for schedule(static)
    for (uint32_t c = 0; c < centroid_amount; ++c)
    {
        if (amount_centroid_points[c] > 0)
        {
            centroids[c] = sum_centroid_points[c] / (float64_t) amount_centroid_points[c];
        }
        else
        {
            centroids[c] = points[0];
        }
    }

    free(thread_sums);
    free(thread_counts);
}

static void kmeans_1d(void)
{
    float64_t sse_holder = 1e300;
    float64_t relative_change = 0.0f;

    sum_centroid_points = calloc((size_t) centroid_amount, sizeof(float64_t));
    amount_centroid_points = calloc((size_t) centroid_amount, sizeof(uint32_t));
    if (!sum_centroid_points || !amount_centroid_points)
    {
        perror("calloc");
        exit(1);
    }

    for (iteration_counter = 0; iteration_counter < iteration_limit; ++iteration_counter)
    {
        assignment_step_1d();
        relative_change = fabs(sum_squared_errors - sse_holder) / (sse_holder > 0.0 ? sse_holder : 1.0);

        if (relative_change < epsilon)
        {
            ++iteration_counter;
            break;
        }
        else
        {
            update_step_1d();
            sse_holder = sum_squared_errors;
        }
    }

    free(sum_centroid_points);
    free(amount_centroid_points);
}

int main(int argc,
         char **argv)
{
    if (argc < 3)
    {
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [num_threads=4]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }

    const char *const path_points = argv[1];
    const char *const path_centroids = argv[2];
    iteration_limit = (argc > 3) ? (uint32_t) atoi(argv[3]) : 50u;
    epsilon = (argc > 4) ? atof(argv[4]) : 1e-4;
    const char *path_assignment = (argc > 5) ? argv[5] : NULL;
    const char *path_output_centroid = (argc > 6) ? argv[6] : NULL;
    int num_threads = (argc > 7) ? atoi(argv[7]) : 4;

    if (iteration_limit <= 0 || epsilon <= 0.0)
    {
        fprintf(stderr, "max_iter>0 e eps>0\n");
        return 1;
    }

    /* Configurar número de threads OpenMP */
    omp_set_num_threads(num_threads);

    points = read_csv_1col(path_points, &point_amount);
    centroids = read_csv_1col(path_centroids, &centroid_amount);
    assignments = malloc((size_t) point_amount * sizeof(uint32_t));
    if (!assignments)
    {
        perror("malloc");
        free(centroids);
        free((void *) points);
        return 1;
    }

    /* Medição de tempo com OpenMP */
    double start_time = omp_get_wtime();

    kmeans_1d();

    double end_time = omp_get_wtime();
    double elapsed_ms = (end_time - start_time) * 1000.0;

    printf("K-means 1D (OpenMP)\n");
    printf("N=%u K=%u max_iter=%u eps=%g threads=%d\n",
           point_amount, centroid_amount, iteration_limit, epsilon, num_threads);
    printf("Iterações: %u | SSE final: %.10f | Tempo: %.3f ms\n",
           iteration_counter, sum_squared_errors, elapsed_ms);

    write_assign_csv(path_assignment, assignments, point_amount);
    write_centroids_csv(path_output_centroid, centroids, centroid_amount);

    free(assignments);
    free(centroids);
    free((void *) points);
    return 0;
}
