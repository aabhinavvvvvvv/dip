/*
 * Q1: Ping-Pong message transfer using MPI_Send and MPI_Recv.
 *
 * Two processes repeatedly exchange messages, doubling the size each
 * time from 64 bytes until 10,000,000 bytes.
 *
 * Deadlock-free ordering:
 *   Rank 0  →  Send first, then Recv
 *   Rank 1  →  Recv first, then Send
 *
 * If both ranks Send before Recv (as shown in the commented block below
 * for rank 1), both block on MPI_Send waiting for a receiver → DEADLOCK.
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char* argv[]) {
    int numprocs, rank, tag = 100, msg_size = 64;
    char* buf;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (numprocs != 2) {
        printf("The number of processes must be two!\n");
        MPI_Finalize();
        return 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("MPI process %d started...\n", rank);
    fflush(stdout);

    while (msg_size < 10000000) {
        msg_size = msg_size * 2;
        buf = (char*)malloc(msg_size * sizeof(char));

        if (rank == 0) {
            MPI_Send(buf, msg_size, MPI_BYTE, rank + 1, tag, MPI_COMM_WORLD);
            printf("Message of length %d to process %d\n", msg_size, rank + 1);
            fflush(stdout);
            MPI_Recv(buf, msg_size, MPI_BYTE, rank + 1, tag, MPI_COMM_WORLD,
                     &status);
        }

        if (rank == 1) {
            /*
             * DEADLOCK-PRONE ordering (both ranks send before receiving):
             *
             * MPI_Send(buf, msg_size, MPI_BYTE, rank-1, tag, MPI_COMM_WORLD);
             * MPI_Recv(buf, msg_size, MPI_BYTE, rank-1, tag, MPI_COMM_WORLD,
             *          &status);
             */

            /* Deadlock-free: Recv first, then Send */
            MPI_Recv(buf, msg_size, MPI_BYTE, rank - 1, tag, MPI_COMM_WORLD,
                     &status);
            MPI_Send(buf, msg_size, MPI_BYTE, rank - 1, tag, MPI_COMM_WORLD);
            printf("Message of length %d to process %d\n", msg_size, rank - 1);
            fflush(stdout);
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}
