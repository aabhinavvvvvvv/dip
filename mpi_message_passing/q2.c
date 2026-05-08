/*
 * Q2: Circular message exchange among 3 processes.
 *
 * Ring topology:  Process 0 → Process 1 → Process 2 → Process 0
 *
 * Message size doubles each iteration starting from 1 byte until it
 * reaches 1,000,000 bytes.
 *
 * Deadlock-free ordering (one process must recv before the ring can move):
 *   Process 0  →  Send to 1,   then Recv from 2
 *   Process 1  →  Recv from 0, then Send to 2
 *   Process 2  →  Recv from 1, then Send to 0
 *
 * If all three send before receiving, every MPI_Send blocks waiting for
 * a matching Recv → circular deadlock.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char* argv[]) {
    int numprocs, rank, tag = 200;
    int msg_size = 1;
    char* buf;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs != 3) {
        if (rank == 0)
            printf("This program requires exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("MPI process %d started\n", rank);
    fflush(stdout);

    int next = (rank + 1) % 3;   /* neighbour to send to   */
    int prev = (rank + 2) % 3;   /* neighbour to recv from */

    while (msg_size <= 1000000) {
        buf = (char*)malloc(msg_size * sizeof(char));
        memset(buf, rank, msg_size);    /* fill with sender rank for verification */

        if (rank == 0) {
            /* Send first so the ring starts moving */
            MPI_Send(buf, msg_size, MPI_BYTE, next, tag, MPI_COMM_WORLD);
            printf("Process 0: sent  %7d bytes to   Process 1\n", msg_size);
            fflush(stdout);

            MPI_Recv(buf, msg_size, MPI_BYTE, prev, tag, MPI_COMM_WORLD, &status);
            printf("Process 0: recvd %7d bytes from Process 2\n", msg_size);
            fflush(stdout);
        } else {
            /* Processes 1 and 2 receive first to avoid deadlock */
            MPI_Recv(buf, msg_size, MPI_BYTE, prev, tag, MPI_COMM_WORLD, &status);
            printf("Process %d: recvd %7d bytes from Process %d\n",
                   rank, msg_size, prev);
            fflush(stdout);

            MPI_Send(buf, msg_size, MPI_BYTE, next, tag, MPI_COMM_WORLD);
            printf("Process %d: sent  %7d bytes to   Process %d\n",
                   rank, msg_size, next);
            fflush(stdout);
        }

        free(buf);
        msg_size *= 2;
    }

    MPI_Finalize();
    return 0;
}
