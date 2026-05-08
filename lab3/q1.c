/*
 * Q1: 5 threads each compute the square of their assigned number (1–5).
 * Each thread dynamically allocates memory for the result and returns it.
 * Main collects results via pthread_join, sums them, and frees the memory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5

void* compute_square(void* arg) {
    int n = *(int*)arg;
    int* result = malloc(sizeof(int));
    *result = n * n;
    printf("Thread %d: %d^2 = %d\n", n, n, *result);
    return result;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int inputs[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        inputs[i] = i + 1;
        pthread_create(&threads[i], NULL, compute_square, &inputs[i]);
    }

    int total = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        int* result;
        pthread_join(threads[i], (void**)&result);
        total += *result;
        free(result);
    }

    printf("\nSum of squares (1^2 + 2^2 + ... + 5^2) = %d\n", total);
    return 0;
}
