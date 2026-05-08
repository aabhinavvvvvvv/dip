/*
 * Floyd-Warshall All-Pairs Shortest Path using OpenMP Tasks
 *
 * Parallelism strategy:
 *   - Outer loop over k runs sequentially (each k depends on k-1 results).
 *   - For each k, a single thread spawns tasks for every (i,j) pair.
 *   - #pragma omp taskwait after each k ensures all updates finish
 *     before the next intermediate vertex is processed.
 *
 * Why no race condition:
 *   At step k, dist[i][k] and dist[k][j] are never modified by any task
 *   (updating dist[k][j] would compute min(dist[k][j], dist[k][k]+dist[k][j])
 *   = dist[k][j] since dist[k][k]=0), so reads are safe without locking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define INF 1e9
#define MAX_V 100

int V;
double dist[MAX_V][MAX_V];

void floyd_warshall_parallel() {
    for (int k = 0; k < V; k++) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 0; i < V; i++) {
                    for (int j = 0; j < V; j++) {
                        #pragma omp task firstprivate(i, j)
                        {
                            double through_k = dist[i][k] + dist[k][j];
                            if (through_k < dist[i][j])
                                dist[i][j] = through_k;
                        }
                    }
                }
                #pragma omp taskwait
            }
        }
    }
}

void floyd_warshall_serial() {
    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++) {
                double through_k = dist[i][k] + dist[k][j];
                if (through_k < dist[i][j])
                    dist[i][j] = through_k;
            }
}

void print_matrix(const char *label) {
    printf("\n%s:\n", label);
    printf("     ");
    for (int j = 0; j < V; j++) printf("%6d", j);
    printf("\n     ");
    for (int j = 0; j < V; j++) printf("------");
    printf("\n");
    for (int i = 0; i < V; i++) {
        printf("%3d |", i);
        for (int j = 0; j < V; j++) {
            if (dist[i][j] >= INF)
                printf("   INF");
            else
                printf("%6.0f", dist[i][j]);
        }
        printf("\n");
    }
}

void load_sample_graph() {
    /*
     * Sample weighted directed graph (4 vertices):
     *
     *   0 --3--> 1
     *   0 --7--> 3
     *   1 --2--> 2
     *   2 --(-)4--> 0   (negative weight, valid for Floyd-Warshall)
     *   3 --1--> 1
     *   3 --2--> 2
     *
     * Expected shortest paths:
     *   0->2 = 5  (0->1->2)
     *   3->0 = -1 (3->1->2->0)
     */
    V = 4;
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++)
            dist[i][j] = (i == j) ? 0 : INF;

    dist[0][1] =  3;
    dist[0][3] =  7;
    dist[1][2] =  2;
    dist[2][0] = -4;
    dist[3][1] =  1;
    dist[3][2] =  2;
}

int main(int argc, char *argv[]) {
    int use_parallel = 1;   /* default: parallel */

    if (argc > 1 && argv[1][0] == 's')
        use_parallel = 0;   /* pass 's' to run serial version */

    load_sample_graph();
    print_matrix("Initial distance matrix");

    double t_start = omp_get_wtime();

    if (use_parallel) {
        printf("\nRunning PARALLEL Floyd-Warshall with OpenMP tasks...\n");
        floyd_warshall_parallel();
    } else {
        printf("\nRunning SERIAL Floyd-Warshall...\n");
        floyd_warshall_serial();
    }

    double t_end = omp_get_wtime();

    print_matrix("All-pairs shortest path matrix");
    printf("\nTime taken : %.6f seconds\n", t_end - t_start);
    printf("OMP threads: %d\n", omp_get_max_threads());

    return 0;
}
