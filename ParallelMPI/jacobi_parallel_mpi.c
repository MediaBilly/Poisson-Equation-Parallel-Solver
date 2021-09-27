#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NORTH 0
#define EAST 1
#define SOUTH 2
#define WEST 3

void Build_input_mpi_type(int *n, int *m, int *mits, double *alpha, double *tol, double *relax, MPI_Datatype* input_mpi_t_p) {
    int array_of_blocklengths[6] = { 1, 1, 1, 1, 1, 1 };
    MPI_Datatype array_of_types[6] = { MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Aint n_addr, m_addr, mits_addr, alpha_addr, tol_addr, relax_addr;
    MPI_Aint array_of_displacements[6] = {0};

    // Get MPI addresses
    MPI_Get_address(n, &n_addr);
    MPI_Get_address(m, &m_addr);
    MPI_Get_address(mits, &mits_addr);
    MPI_Get_address(alpha, &alpha_addr);
    MPI_Get_address(tol, &tol_addr);
    MPI_Get_address(relax, &relax_addr);

    // Calculate displacements
    array_of_displacements[1] = m_addr - n_addr;
    array_of_displacements[2] = mits_addr - n_addr;
    array_of_displacements[3] = alpha_addr - n_addr;
    array_of_displacements[4] = tol_addr - n_addr;
    array_of_displacements[5] = relax_addr - n_addr;

    // Create the custom datatype
    MPI_Type_create_struct(6, array_of_blocklengths, array_of_displacements, array_of_types, input_mpi_t_p);
    MPI_Type_commit(input_mpi_t_p);
}

void Get_input(int my_rank, int comm_sz, int *n, int *m, int *mits, double *alpha, double *tol, double *relax) {
    // Build custom MPI datatype for the input
    MPI_Datatype input_mpi_t;
    Build_input_mpi_type(n, m, mits, alpha, tol, relax, &input_mpi_t);

    // Read input from stdio
    if (my_rank == 0) {
        
        scanf("%d,%d", n, m);
        scanf("%lf", alpha);
        scanf("%lf", relax);
        scanf("%lf", tol);
        scanf("%d", mits);
    } 

    // Broadcast it to all the children
    MPI_Bcast(n, 1, input_mpi_t, 0, MPI_COMM_WORLD);

    // Free it's apace
    MPI_Type_free(&input_mpi_t);
}

int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;

    int totalProcesses, myRank;
    int dim;
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get number of total processes
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    // Get rank of current process
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    // Check if it is valid for the current problem(perfect square)
    dim = (int)sqrt(totalProcesses);
    if (dim * dim != totalProcesses) {
        if (myRank == 0) {
            fprintf(stderr, "Given # of processes is not a perfect square.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Read input
    Get_input(myRank, totalProcesses, &n, &m, &mits, &alpha, &tol, &relax);

    // Create cartesian topology
    int ndims = 2, reorder = 1, periods[2], dimSize[2];
    MPI_Comm cartComm;
    dimSize[0] = dim;
    dimSize[1] = dim;
    periods[0] = periods[1] = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dimSize, periods, reorder, &cartComm);

    // Get rank of current process
    MPI_Comm_rank(cartComm, &myRank);
    // Get coords of current process
    int my_coords[2];
    MPI_Cart_coords(cartComm, myRank, ndims, my_coords);

    // Find neighbours of current process
    int neighbours[4];
    // West and east
    MPI_Cart_shift(cartComm, 1, 1, &neighbours[WEST], &neighbours[EAST]);
    // North and south 
    MPI_Cart_shift(cartComm, 0, 1, &neighbours[NORTH], &neighbours[SOUTH]);
    printf("Process %d(%d, %d) of %d --> NORTH: %d, SOUTH: %d, WEST: %d, EAST: %d, ARGUMENTS: %d, %d, %d, %lf, %lf, %lf\n", myRank, my_coords[0], my_coords[1], totalProcesses, neighbours[NORTH], neighbours[SOUTH], neighbours[WEST], neighbours[EAST], n, m, mits, alpha, tol, relax);

    // Close MPI
    MPI_Finalize();
    return 0;
}
