#include <stdio.h>
#include <pthread.h>

void* myThread(void* arg) {
    printf("Hello from the new thread!\n");
    return NULL;
}

int main() {
    pthread_t thread;
    pthread_create(&thread, NULL, myThread, NULL);
    pthread_join(thread, NULL); // Wait for thread to finish
    return 0;
}
