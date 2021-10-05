#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

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

/**********************************************************
 * Calculates the square error between numerical and exact local solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double fX, fY;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        fY = yStart + (y-1)*deltaY;
        for (x = 1; x < (maxXCount-1); x++)
        {
            fX = xStart + (x-1)*deltaX;
            localError = U(x,y) - (1.0-fX*fX)*(1.0-fY*fY);
            error += localError*localError;
        }
    }
    return error;
}

int main(int argc, char **argv)
{
    // Global variables
    unsigned int n, m, mits;
    double alpha, tol, relax;
    double square_error, absolute_square_error;
    double residual; // Used if convergence check is on
    double totalTime;
    unsigned int totalProcesses, myRank;
    unsigned int dim;

    // Parse command line argumnets
    unsigned int convergence_check = (argc == 2 && !strcmp(argv[1], "-c"));

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Pcontrol(0);

    // Get number of total processes
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);
    // Get rank of current process
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    // Check if it is valid for the current problem(perfect square)
    // dim is the # of blocks(processes) per row or column
    dim = (int)sqrt(totalProcesses);

    if (totalProcesses != 80 && dim * dim != totalProcesses) {
        if (myRank == 0) {
            fprintf(stderr, "Given # of processes is not a perfect square.\n");
        }

        MPI_Finalize();
        return 0;
    }

    // Read input
    Get_input(myRank, totalProcesses, &n, &m, &mits, &alpha, &tol, &relax);

    if (myRank == 0) {
        printf("Grid size: %dx%d\nProcesses: %d\n", n, m, totalProcesses);
    }

    // Create cartesian topology
    unsigned int ndims = 2, reorder = 1, periods[2], dimSize[2];
    MPI_Comm cartComm;

    if(totalProcesses != 80) {
        dimSize[0] = dim;
        dimSize[1] = dim;
    }
    else {
        dimSize[0] = 10;
        dimSize[1] = totalProcesses / dimSize[0];
    }
    
    periods[0] = periods[1] = 0;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dimSize, periods, reorder, &cartComm);

    // Get rank of current process
    MPI_Comm_rank(cartComm, &myRank);
    // Get coords of current process
    unsigned int my_coords[2];
    MPI_Cart_coords(cartComm, myRank, ndims, my_coords);

    // Find neighbours of current process
    int neighbours[4];
    // West and east
    MPI_Cart_shift(cartComm, 1, 1, &neighbours[WEST], &neighbours[EAST]);
    // North and south 
    MPI_Cart_shift(cartComm, 0, 1, &neighbours[NORTH], &neighbours[SOUTH]);
    // printf("Process %d(%d, %d) of %d --> NORTH: %d, SOUTH: %d, WEST: %d, EAST: %d, ARGUMENTS: %d, %d, %d, %lf, %lf, %lf\n", myRank, my_coords[0], my_coords[1], totalProcesses, neighbours[NORTH], neighbours[SOUTH], neighbours[WEST], neighbours[EAST], n, m, mits, alpha, tol, relax);

    // Create subgrid on each process
    unsigned int columns = n/dimSize[0], rows = m/dimSize[1];
    double *u, *u_old, *tmp;
    unsigned int maxXcount = columns+2, maxYcount = rows+2;
    unsigned int allocCount = maxXcount*maxYcount;
    u = (double*)calloc(allocCount, sizeof(double));
    u_old = (double*)calloc(allocCount, sizeof(double));

    if (u == NULL || u_old == NULL) {
        printf("Not enough memory for two %ix%i matrices\n", maxXcount, maxYcount);
        exit(1);
    }

    #define SRC(XX,YY) u_old[(YY)*maxXcount+(XX)]
    #define DST(XX,YY) u[(YY)*maxXcount+(XX)]

    // Create column datatype
    MPI_Datatype col_t;
    MPI_Type_vector(rows, 1, maxXcount, MPI_DOUBLE, &col_t);
    MPI_Type_commit(&col_t);

    // Create row datatype
    MPI_Datatype row_t;
    MPI_Type_contiguous(columns, MPI_DOUBLE, &row_t);
    MPI_Type_commit(&row_t);

    // Wait for all processes to initialize their data
    MPI_Barrier(cartComm);

    // Main loop
    unsigned int iterationCount, maxIterationCount = mits;
    double maxAcceptableError = tol;

    // [-1,1] x [-1,1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    double xStart = xLeft + my_coords[1] * columns * deltaX;
    double yStart = yUp - (my_coords[0] + 1) * rows * deltaY;

    // printf("Process %d (%d, %d) dimension:(%d,%d), start coords: (%lf, %lf)\n", myRank, my_coords[0], my_coords[1], columns, rows, xStart, yStart);

    // Coefficients
    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;

    iterationCount = 0;
    double local_square_error = HUGE_VAL;

    double t1 = MPI_Wtime();
    MPI_Pcontrol(1);

    double *fx_thing_buf = (double*)malloc((columns+2)*sizeof(double));
    /* Iterate as long as it takes to meet the convergence criterion */
    while (iterationCount < maxIterationCount) { 
        MPI_Request send_requests[4], receive_requests[4];
        MPI_Status send_status[4], receive_status[4];
        // Non-blocking Isend of the 4 outer rows and columns to their corresponding neighbours (from u_old)

        // North
        MPI_Isend(&SRC(1, 1), 1, row_t, neighbours[NORTH], 0, cartComm, &send_requests[NORTH]);
        // South
        MPI_Isend(&SRC(1, rows), 1, row_t, neighbours[SOUTH], 0, cartComm, &send_requests[SOUTH]);
        // West
        MPI_Isend(&SRC(1, 1), 1, col_t, neighbours[WEST], 0, cartComm, &send_requests[WEST]);
        // East
        MPI_Isend(&SRC(columns, 1), 1, col_t, neighbours[EAST], 0, cartComm, &send_requests[EAST]);

        // Non-blocking Irecv of the 4 halo rows and columns from their corresponding neighbours

        // North
        MPI_Irecv(&SRC(1, 0), 1, row_t, neighbours[NORTH], 0, cartComm, &receive_requests[NORTH]);
        // South
        MPI_Irecv(&SRC(1, rows + 1), 1, row_t, neighbours[SOUTH], 0, cartComm, &receive_requests[SOUTH]);
        // West
        MPI_Irecv(&SRC(0, 1), 1, col_t, neighbours[WEST], 0, cartComm, &receive_requests[WEST]);
        // East
        MPI_Irecv(&SRC(columns+1, 1), 1, col_t, neighbours[EAST], 0, cartComm, &receive_requests[EAST]);

        // Calculate inner local values (to u)

        unsigned int x, y;
        double fX, fY, fy_thing;
        local_square_error = 0.0;
        double updateVal;
        double f;

        #pragma omp parallel for reduction(+:local_square_error) collapse(2)
        for (y = 2; y < (maxYcount-2); y++)
        {
            for (x = 2; x < (maxXcount-2); x++)
            {
                fY = yStart + (y-1)*deltaY;
                fy_thing = 1.0-fY*fY;
                if(y == 2 && iterationCount == 0) {
                    fX = xStart + (x-1)*deltaX;
                    fx_thing_buf[x-1] = 1.0-fX*fX;
                }
                
                f = -alpha*fx_thing_buf[x-1]*fy_thing - 2.0*fx_thing_buf[x-1] - 2.0*fy_thing;

                updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                                (SRC(x,y-1) + SRC(x,y+1))*cy +
                                SRC(x,y)*cc - f
                            )/cc;

                DST(x,y) = SRC(x,y) - relax*updateVal;
                local_square_error += updateVal*updateVal;
            }
            
        }
        // Wait for the 4 halos to be received
        MPI_Waitall(4, receive_requests, receive_status);

        // Calculate the values of the 4 outer rows and columns which depend on the received halos (to u)
        // Calculate 1st horizontal row
        fY = yStart;
        fy_thing = 1.0-fY*fY;

        #pragma omp parallel for reduction (+:local_square_error)
        for (x = 1; x < (maxXcount-1); x++)
        {            
            f = -alpha*fx_thing_buf[x-1]*fy_thing - 2.0*fx_thing_buf[x-1] - 2.0*fy_thing;

            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                            (SRC(x,y-1) + SRC(x,y+1))*cy +
                            SRC(x,y)*cc - f
                        )/cc;

            DST(x,y) = SRC(x,y) - relax*updateVal;
            local_square_error += updateVal*updateVal;
        }
        
        // Calculate verical columns
        // #pragma omp parallel for reduction (+:local_square_error)
        for (y = 2; y < (maxYcount-2); y++)
        {
            fY = yStart + (y-1)*deltaY;
            fy_thing = 1.0-fY*fY;
            
            for (x = 1; x < (maxXcount-1); x += maxXcount-3)
            {
                if(y == 2 && iterationCount == 0) {
                    fX = xStart + (x-1)*deltaX;
                    fx_thing_buf[x-1] = 1.0-fX*fX;
                }
                
                f = -alpha*fx_thing_buf[x-1]*fy_thing - 2.0*fx_thing_buf[x-1] - 2.0*fy_thing;

                updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                                (SRC(x,y-1) + SRC(x,y+1))*cy +
                                SRC(x,y)*cc - f
                            )/cc;

                DST(x,y) = SRC(x,y) - relax*updateVal;
                local_square_error += updateVal*updateVal;
            }
        }

        // Calculate last horizontal row
        fY = yStart + (maxYcount-3)*deltaY;
        fy_thing = 1.0-fY*fY;

        #pragma omp parallel for reduction(+:local_square_error)
        for (int x = 1; x < (maxXcount-1); x++)
        {
            f = -alpha*fx_thing_buf[x-1]*fy_thing - 2.0*fx_thing_buf[x-1] - 2.0*fy_thing;

            updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                            (SRC(x,y-1) + SRC(x,y+1))*cy +
                            SRC(x,y)*cc - f
                        )/cc;

            DST(x,y) = SRC(x,y) - relax*updateVal;
            local_square_error += updateVal*updateVal;
        }

        // Wait for the 4 outer rows to be sent
        MPI_Waitall(4, send_requests, send_status);

        // Swap the buffers
        tmp = u_old;
        u_old = u;
        u = tmp;

        // Increase iteration count
        iterationCount++;

        if (convergence_check) {
            MPI_Allreduce(&local_square_error, &residual, 1, MPI_DOUBLE, MPI_SUM, cartComm);
            if (sqrt(residual)/(n * m) <= maxAcceptableError) {
                break;
            }
        }
    }

    double t2 = MPI_Wtime();
    MPI_Pcontrol(0);

    double localTime = t2 - t1;
    // Calculate total time(max time of all processes)
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, cartComm);
    if (myRank == 0) {
        printf("Elapsed MPI Wall time: %f\n", totalTime);
    }

    // Print iterations taken
    if (myRank == 0) {
        printf("Total iterations: %d\n", iterationCount);
    }

    // Calculate total residual
    MPI_Reduce(&local_square_error, &square_error, 1, MPI_DOUBLE, MPI_SUM, 0, cartComm);
    if (myRank == 0) {
        printf("Residual %g\n",sqrt(square_error)/(n * m));
    }

    // Calculate total absolute error
    double local_absolute_square_error = checkSolution(xStart, yStart, maxXcount, maxYcount, u, deltaX, deltaY, alpha);
    MPI_Reduce(&local_absolute_square_error, &absolute_square_error, 1, MPI_DOUBLE, MPI_SUM, 0, cartComm);
    if (myRank == 0) {
        printf("The error of the iterative solution is %g\n", sqrt(absolute_square_error)/(n * m));
    }

    // Free data types
    MPI_Type_free(&row_t);
    MPI_Type_free(&col_t);

    // Free allocated space
    free(u);
    free(u_old);
    free(fx_thing_buf);

    // Close MPI
    MPI_Finalize();
    return 0;
}
