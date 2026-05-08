#include <stdio.h>
#include <pthread.h>

// Thread function
void* thread_main(void* arg) {
    printf("Thread is running.\n");
    pthread_exit((void*) "Thread finished"); // Explicit termination
}

int main() {
    pthread_t thread;
    void* exit_status;

    // Create a thread
    pthread_create(&thread, NULL, thread_main, NULL);

    // Wait for the thread to finish and retrieve its exit status
    pthread_join(thread, &exit_status);

    printf("Thread exited with message: %s\n", (char*)exit_status);
    return 0;
}
