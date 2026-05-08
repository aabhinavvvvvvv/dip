/*
 * Q2: 10 threads each compute the factorial of the prime number at their
 * assigned index. Results are dynamically allocated and returned to main,
 * which sums all factorials and prints the total.
 *
 * Each thread receives its own index copy to prevent race conditions from
 * sharing a loop variable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 10

int primes[NUM_THREADS] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};

unsigned long long factorial(int n) {
    unsigned long long f = 1;
    for (int i = 2; i <= n; i++)
        f *= i;
    return f;
}

void* compute_factorial(void* arg) {
    int idx = *(int*)arg;
    int p = primes[idx];
    unsigned long long* result = malloc(sizeof(unsigned long long));
    *result = factorial(p);
    printf("Thread %2d: %2d! = %llu\n", idx, p, *result);
    return result;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int indices[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        indices[i] = i;
        pthread_create(&threads[i], NULL, compute_factorial, &indices[i]);
    }

    unsigned long long total = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        unsigned long long* result;
        pthread_join(threads[i], (void**)&result);
        total += *result;
        free(result);
    }

    printf("\nSum of factorials of first 10 primes = %llu\n", total);
    return 0;
}
