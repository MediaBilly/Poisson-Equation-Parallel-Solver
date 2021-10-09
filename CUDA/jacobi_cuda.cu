#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256 // Number of threads in each block

/**********************************************************
 * Checks the error between numerical and exact solutions
 **********************************************************/
double checkSolution(double xStart, double yStart,
                     int maxXCount, int maxYCount,
                     double *u,
                     double deltaX, double deltaY,
                     double alpha, double *fx_thing, double *fy_thing)
{
#define U(XX,YY) u[(YY)*maxXCount+(XX)]
    int x, y;
    double localError, error = 0.0;

    for (y = 1; y < (maxYCount-1); y++)
    {
        for (x = 1; x < (maxXCount-1); x++)
        {
            localError = U(x,y) - fx_thing[x-1]*fy_thing[y-1];
            error += localError*localError;
        }
    }
    return sqrt(error)/((maxXCount-2)*(maxYCount-2));
}

__global__ void jacobiIteration(int n, int m, double alpha, double relax, double cx, double cy, double cc, double *u, double *u_old, double *fx_thing, double *fy_thing) {
    #define SRC(XX,YY) u_old[(YY)*(n+2)+(XX)]
    #define DST(XX,YY) u[(YY)*(n+2)+(XX)]
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x < n+1 && y < n+1) {
        double f = -alpha*fx_thing[x-1]*fy_thing[y-1] - 2.0*fx_thing[x-1] - 2.0*fy_thing[y-1];
        double updateVal = (	(SRC(x-1,y) + SRC(x+1,y))*cx +
                        (SRC(x,y-1) + SRC(x,y+1))*cy +
                        SRC(x,y)*cc - f
                    )/cc;
        DST(x,y) = SRC(x,y) - relax*updateVal;
    }
}


int main(int argc, char **argv)
{
    int n, m, mits;
    double alpha, tol, relax;
    // double maxAcceptableError;
    // double error;
    int allocCount;
    int iterationCount, maxIterationCount;
    // double t1, t2;

//    printf("Input n,m - grid dimension in x,y direction:\n");
    scanf("%d,%d", &n, &m);
//    printf("Input alpha - Helmholtz constant:\n");
    scanf("%lf", &alpha);
//    printf("Input relax - successive over-relaxation parameter:\n");
    scanf("%lf", &relax);
//    printf("Input tol - error tolerance for the iterrative solver:\n");
    scanf("%lf", &tol);
//    printf("Input mits - maximum solver iterations:\n");
    scanf("%d", &mits);

    allocCount = (n+2)*(m+2);

    // Solve in [-1, 1] x [-1, 1]
    double xLeft = -1.0, xRight = 1.0;
    double yBottom = -1.0, yUp = 1.0;

    double deltaX = (xRight-xLeft)/(n-1);
    double deltaY = (yUp-yBottom)/(m-1);

    double cx = 1.0/(deltaX*deltaX);
    double cy = 1.0/(deltaY*deltaY);
    double cc = -2.0*cx-2.0*cy-alpha;

    // Allocate memory for pre-calculated stuff for the GPU
    double *fx_thing, *fy_thing;
    cudaMallocManaged(&fx_thing, n*sizeof(double));
    cudaMallocManaged(&fy_thing, m*sizeof(double));

    // Precalucate stuff to save time

    int x,y;
    for (x = 1; x < n+1; x++) {
        double fX = xLeft + (x-1)*deltaX;
        fx_thing[x-1] = 1.0-fX*fX;
    }

    for (y = 1; y < m+1; y++) {
        double fY = yBottom + (y-1)*deltaY;
        fy_thing[y-1] = 1.0-fY*fY;
    }

    // Allocate u and u_old for GPU
    double *d_u, *d_u_old;
    cudaMalloc(&d_u, allocCount * sizeof(double));
    cudaMalloc(&d_u_old, allocCount * sizeof(double));
    cudaMemset(d_u, 0, allocCount * sizeof(double));
    cudaMemset(d_u_old, 0, allocCount * sizeof(double));

    // Calculate GridSize
    dim3 gridSize(ceil(n/sqrt(BLOCK_SIZE)), ceil(m/sqrt(BLOCK_SIZE)));

    // Calculate Block size
    dim3 blockSize(sqrt(BLOCK_SIZE), sqrt(BLOCK_SIZE));

    printf("Grid size: %dx%d\n\n", n, m);

    //Run main loop
    iterationCount = 0;
    maxIterationCount = mits;
    double *tmp;
    // maxAcceptableError = tol;
    double t1 = clock();
    while (iterationCount < maxIterationCount) {
        jacobiIteration<<<gridSize, blockSize>>>(n, m, alpha, relax, cx, cy, cc, d_u, d_u_old, fx_thing, fy_thing);

        cudaDeviceSynchronize();

        iterationCount++;

        // Swap the buffers
        tmp = d_u_old;
        d_u_old = d_u;
        d_u = tmp;
    }
    double t2 = clock();
    printf( "Iterations=%3d Elapsed time is %f\n", iterationCount, (double)(t2 - t1)/CLOCKS_PER_SEC);

    // Copy grid to host
    double *u_old = (double*)malloc(allocCount * sizeof(double));
    cudaMemcpy(u_old, d_u_old, allocCount * sizeof(double), cudaMemcpyDeviceToHost);

    // u_old holds the solution after the most recent buffers swap
    double absoluteError = checkSolution(xLeft, yBottom,
                                         n+2, m+2,
                                         u_old,
                                         deltaX, deltaY,
                                         alpha, fx_thing, fy_thing);
    printf("The error of the iterative solution is %g\n", absoluteError);
    

    // Free GPU memory
    cudaFree(fy_thing);
    cudaFree(fx_thing);
    cudaFree(d_u_old);
    cudaFree(d_u);

    // Free host memory
    free(u_old);
}