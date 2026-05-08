/*
 * Q3: 10 threads each check whether their assigned array element is a perfect
 * number (equals the sum of its proper divisors).
 *
 * If perfect: dynamically allocate an int, store the value, return it.
 * If not:     return NULL.
 *
 * A shared global counter tracks perfect numbers found; a mutex guards it
 * to prevent race conditions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 10

int arr[NUM_THREADS] = {1, 6, 10, 28, 15, 496, 20, 100, 25, 8128};

int perfect_count = 0;
pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;

int is_perfect(int n) {
    if (n < 2) return 0;
    int sum = 1;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i)
                sum += n / i;
        }
    }
    return sum == n;
}

void* check_perfect(void* arg) {
    int idx = *(int*)arg;
    int val = arr[idx];

    if (is_perfect(val)) {
        pthread_mutex_lock(&count_mutex);
        perfect_count++;
        pthread_mutex_unlock(&count_mutex);

        int* result = malloc(sizeof(int));
        *result = val;
        printf("Thread %2d: %d is a PERFECT number\n", idx, val);
        return result;
    }

    printf("Thread %2d: %d is not perfect\n", idx, val);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int indices[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        indices[i] = i;
        pthread_create(&threads[i], NULL, check_perfect, &indices[i]);
    }

    printf("\n--- Perfect numbers found ---\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        int* result;
        pthread_join(threads[i], (void**)&result);
        if (result != NULL) {
            printf("  %d\n", *result);
            free(result);
        }
    }

    printf("\nTotal perfect numbers found: %d\n", perfect_count);

    pthread_mutex_destroy(&count_mutex);
    return 0;
}
