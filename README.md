# Poisson-Equation
A parallel algorithm to solve Poisson equation using Jacobi with Successive Over-Relaxation(S.O.R) algorithm.

The goal of this project was to develop a parallel program that numerically solves Poisson's equation:

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20%28%7B%5Cnabla%7D%5E2%20-%20a%29u%3D%5Cfrac%7Bd%5E2%7D%7Bdx%5E2%7Du%20&plus;%20%5Cfrac%7Bd%5E2%7D%7Bdy%5E2%7Du%20-%20au%20%3D%20f)

in range [-1,1]x[-1,1].

The program calculates the values of the matrix `u` in the range above and also outputs the error between the numerical and the analytical solution which is:

![equation](https://latex.codecogs.com/png.latex?%5Cbg_white%20u%28x%2Cy%29%20%3D%20%281-x%5E2%29%281-y%5E2%29)

The libraries we used to parallelize the algorithm are:
  - MPI
  - Hybrid MPI+OpenMp
  - CUDA
  
## Execution Instructions
  - To run the MPI program, navigate to the `ParallelMPI` folder, compile using `make` and run with `mpirun jacobi_parallel_mpi.x -c < input`
  where `input` is the input file with the parameters(example given in the folder).
  - To run the Hybrid MPI+OpenMp program, navigate to the `HybridMPI` folder, compile using `make` and run with `mpirun --bind-to none jacobi_hybrid_mpi.x -c < input`
  where `input` is the input file with the parameters(example given in the folder).
  - To run the CUDA program, navigate to the `CUDA` folder, compile with make and run with  `./jacobi_cuda.x < input` for 1 GPU or `./jacobi_cuda_2gpus.x < input` for 2 GPUs.
  
NOTE: All the programs come with a pbs script, because the benchmarks were made in a cluster provided by our university called `argo`. To run them on your own system, the process may be different.

## Benchmarks

The benchmarks below, were made in a cluster provided by our university called `argo` which consists of the following nodes:
  * A front-end or master node (Dell OptiPlex 7050) to which we remotely connect to and has the following specs: 
    - Quad-Core Intel-Core i5-6500 @3.20GHz
    - RAM 16GB DDR4 2.4GHz
  * 10 compute nodes (Dell PowerEdge 1950) connected to the master node with ethernet 1G which were used to make the MPI and Hybrid MPI+OpenMP benchmarks. Each one of those has the following specs:
    - Dual XEON E5340 @2.66GHz quad core (total 8 cores each node)
    - RAM 16GB DDR2 667MHz
    - Bus interconnection
    - Connected together with 10G ethernet
  * 1 GPU compute node (Dell Precision 7920 Rack):
    - Intel Xeon Silver 4114 @ 2.2 GHz
    - 10 Cores with hyperthreading
    - RAM 16GB (2X8GB) DDR4 2666MHz 
    - Dual Nvidia Quatro P4000:
      - 1792 CUDA cores
      - RAM 8GB
      - Memory Bandwidth: 243 GB/s
      
Here are the benchmarks we made in each program using the above cluster:

### Sequential Program:

| Grid Size     | 840x840 | 1680x1680 | 3360x3360 | 6720x6720 | 13440x13440 | 26880x26880 |
| ------------- | ------- | --------- | --------- | --------- | ----------- | ----------- |
| Time(seconds) | 0.511   | 1.905     | 7.48      | 29.77     | 118.953     | 477.191     |
| Error         | 0.000633904 | 0.000317239 | 0.000158679 | 7.93528e-05 | 3.96795e-05 | 1.98405e-05 |

### MPI Program without convergence check (no AllReduce calls):

|                       |   |      |      | Timings |       |       |       |       |       |
| --------------------- | - | ---- | ---- | ------- | ----- | ----- | ----- | ----- | ----- |
| Grid Size/ Processes | 1       | 4       | 9      | 16     | 25     | 36     | 49     | 64     | 80     |
| 840x840               | 0.800   | 0.203   | 0.097  | 0.072  | 0.045  | 0.046  | 0.039  | 0.039  | 0.042  |
| 1680x1680             | 3.185   | 0.816   | 0.377  | 0.215  | 0.147  | 0.105  | 0.095  | 0.076  | 0.076  |
| 3360x3360             | 12.700  | 3.241   | 1.474  | 0.933  | 0.561  | 0.411  | 0.299  | 0.227  | 0.194  |
| 6720x6720             | 50.775  | 12.839  | 5.794  | 3.604  | 2.191  | 1.636  | 1.149  | 0.928  | 0.761  |
| 13440x13440           | 202.996 | 51.198  | 23.022 | 14.184 | 8.555  | 6.385  | 4.408  | 3.610  | 2.903  |
| 26880x26880           | 815.197 | 204.764 | 91.834 | 56.695 | 33.995 | 25.227 | 17.460 | 14.304 | 11.420 |


|                       |   |      |      | Speedup |       |       |       |       |       |
| --------------------- | - | ---- | ---- | ------- | ----- | ----- | ----- | ----- | ----- |
| Grid Size/ Processes | 1 | 4    | 9    | 16      | 25    | 36    | 49    | 64    | 80    |
| 840x840               | 1 | 3.94 | 8.21 | 11.04   | 17.77 | 17.50 | 20.47 | 20.46 | 18.91 |
| 1680x1680             | 1 | 3.90 | 8.46 | 14.81   | 21.67 | 30.26 | 33.58 | 42.05 | 41.78 |
| 3360x3360             | 1 | 3.92 | 8.62 | 13.61   | 22.65 | 30.89 | 42.52 | 55.92 | 65.45 |
| 6720x6720             | 1 | 3.95 | 8.76 | 14.09   | 23.17 | 31.04 | 44.17 | 54.72 | 66.71 |
| 13440x13440           | 1 | 3.96 | 8.82 | 14.31   | 23.73 | 31.79 | 46.05 | 56.22 | 69.93 |
| 26880x26880           | 1 | 3.98 | 8.88 | 14.38   | 23.98 | 32.31 | 46.69 | 56.99 | 71.39 |

|                       |   |      |      | Efficiency |      |      |      |      |      |
| --------------------- | - | ---- | ---- | ---------- | ---- | ---- | ---- | ---- | ---- |
| Grid Size/ Processes | 1 | 4    | 9    | 16         | 25   | 36   | 49   | 64   | 80   |
| 840x840               | 1 | 0.99 | 0.91 | 0.69       | 0.71 | 0.49 | 0.42 | 0.32 | 0.24 |
| 1680x1680             | 1 | 0.98 | 0.94 | 0.93       | 0.87 | 0.84 | 0.69 | 0.66 | 0.52 |
| 3360x3360             | 1 | 0.98 | 0.96 | 0.85       | 0.91 | 0.86 | 0.87 | 0.87 | 0.82 |
| 6720x6720             | 1 | 0.99 | 0.97 | 0.88       | 0.93 | 0.86 | 0.90 | 0.85 | 0.83 |
| 13440x13440           | 1 | 0.99 | 0.98 | 0.89       | 0.95 | 0.88 | 0.94 | 0.88 | 0.87 |
| 26880x26880           | 1 | 1.00 | 0.99 | 0.90       | 0.96 | 0.90 | 0.95 | 0.89 | 0.89 |

### MPI Program with convergence check (with AllReduce calls):


|                       |   |      |      | Timings |       |       |       |       |       |
| --------------------- | - | ---- | ---- | ------- | ----- | ----- | ----- | ----- | ----- |
| Grid Size/ Διεργασίες | 1       | 4       | 9      | 16     | 25     | 36     | 49     | 64     | 80     ||
| 840x840               | 0.803   | 0.211   | 0.110  | 0.080  | 0.078  | 0.074  | 0.064  | 0.043  | 0.081  |
| 1680x1680             | 3.183   | 0.826   | 0.376  | 0.234  | 0.166  | 0.155  | 0.128  | 0.111  | 0.108  |
| 3360x3360             | 12.700  | 3.244   | 1.479  | 0.936  | 0.579  | 0.457  | 0.333  | 0.250  | 0.270  |
| 6720x6720             | 50.779  | 12.849  | 5.783  | 3.604  | 2.183  | 1.666  | 1.180  | 0.969  | 0.796  |
| 13440x13440           | 202.993 | 51.211  | 22.956 | 14.189 | 8.546  | 6.393  | 4.440  | 3.623  | 2.930  |
| 26880x26880           | 817.426 | 204.736 | 91.542 | 56.489 | 33.825 | 25.254 | 17.367 | 14.220 | 11.436 |

|                       |   |      |      | Speedup |       |       |       |       |       |
| --------------------- | - | ---- | ---- | ------- | ----- | ----- | ----- | ----- | ----- |
| Grid Size/ Διεργασίες | 1 | 4    | 9    | 16      | 25    | 36    | 49    | 64    | 80    |
| 840x840               | 1 | 3.80 | 7.28 | 9.98    | 10.35 | 10.91 | 12.52 | 18.50 | 9.93  |
| 1680x1680             | 1 | 3.85 | 8.47 | 13.62   | 19.19 | 20.53 | 24.78 | 28.76 | 29.44 |
| 3360x3360             | 1 | 3.92 | 8.59 | 13.57   | 21.92 | 27.79 | 38.12 | 50.80 | 46.98 |
| 6720x6720             | 1 | 3.95 | 8.78 | 14.09   | 23.26 | 30.48 | 43.02 | 52.38 | 63.79 |
| 13440x13440           | 1 | 3.96 | 8.84 | 14.31   | 23.75 | 31.75 | 45.72 | 56.02 | 69.28 |
| 26880x26880           | 1 | 3.99 | 8.93 | 14.47   | 24.17 | 32.37 | 47.07 | 57.48 | 71.48 |

|                       |   |      |      | Efficiency |      |      |      |      |      |
| --------------------- | - | ---- | ---- | ---------- | ---- | ---- | ---- | ---- | ---- |
| Grid Size/ Διεργασίες | 1 | 4    | 9    | 16         | 25   | 36   | 49   | 64   | 80   |
| 840x840               | 1 | 0.95 | 0.81 | 0.62       | 0.41 | 0.30 | 0.26 | 0.29 | 0.12 |
| 1680x1680             | 1 | 0.96 | 0.94 | 0.85       | 0.77 | 0.57 | 0.51 | 0.45 | 0.37 |
| 3360x3360             | 1 | 0.98 | 0.95 | 0.85       | 0.88 | 0.77 | 0.78 | 0.79 | 0.59 |
| 6720x6720             | 1 | 0.99 | 0.98 | 0.88       | 0.93 | 0.85 | 0.88 | 0.82 | 0.80 |
| 13440x13440           | 1 | 0.99 | 0.98 | 0.89       | 0.95 | 0.88 | 0.93 | 0.88 | 0.87 |
| 26880x26880           | 1 | 1.00 | 0.99 | 0.90       | 0.97 | 0.90 | 0.96 | 0.90 | 0.89 |

### Hybrid MPI+OpenMp Program without convergence check (no AllReduce calls):

|                       |   |      |      | Timings |       |       |       | 
| --------------------- | - | ---- | ---- | ------- | ----- | ----- | ----- | 
| Grid Size/ Processes - Threads | 1       | 1p-4t   | 2p-8t   | 4p-16t | 9p-36t | 16p-64t | 20p-80t |
| 840x840                        | 0.800   | 0.205   | 0.106   | 0.071  | 0.037  | 0.041   | 0.068   |
| 1680x1680                      | 3.185   | 0.810   | 0.446   | 0.226  | 0.108  | 0.083   | 0.056   |
| 3360x3360                      | 12.700  | 3.211   | 1.801   | 0.931  | 0.420  | 0.241   | 0.195   |
| 6720x6720                      | 50.775  | 12.820  | 7.091   | 3.563  | 1.655  | 0.948   | 0.776   |
| 13440x13440                    | 202.996 | 51.167  | 28.149  | 14.185 | 6.434  | 3.668   | 2.936   |
| 26880x26880                    | 815.197 | 207.841 | 115.079 | 56.544 | 25.572 | 14.486  | 11.744  |

|                    |   |      |      | Speedup |       |       |       |
| ------------------ | - | ---- | ---- | ------- | ----- | ----- | ----- |
| Grid Size/ Threads | 1 | 4    | 8    | 16      | 36    | 64    | 80    |
| 840x840            | 1 | 3.90 | 7.54 | 11.26   | 21.84 | 19.39 | 11.69 |
| 1680x1680          | 1 | 3.93 | 7.14 | 14.08   | 29.52 | 38.53 | 57.07 |
| 3360x3360          | 1 | 3.96 | 7.05 | 13.65   | 30.25 | 52.67 | 65.26 |
| 6720x6720          | 1 | 3.96 | 7.16 | 14.25   | 30.68 | 53.56 | 65.44 |
| 13440x13440        | 1 | 3.97 | 7.21 | 14.31   | 31.55 | 55.34 | 69.15 |
| 26880x26880        | 1 | 3.92 | 7.08 | 14.42   | 31.88 | 56.27 | 69.42 |

|                    |   |      |      | Efficiency |      |      |      |
| ------------------ | - | ---- | ---- | ---------- | ---- | ---- | ---- |
| Grid Size/ Threads | 1 | 4    | 8    | 16         | 36   | 64   | 80   |
| 840x840            | 1 | 0.98 | 0.94 | 0.70       | 0.61 | 0.30 | 0.15 |
| 1680x1680          | 1 | 0.98 | 0.89 | 0.88       | 0.82 | 0.60 | 0.71 |
| 3360x3360          | 1 | 0.99 | 0.88 | 0.85       | 0.84 | 0.82 | 0.82 |
| 6720x6720          | 1 | 0.99 | 0.90 | 0.89       | 0.85 | 0.84 | 0.82 |
| 13440x13440        | 1 | 0.99 | 0.90 | 0.89       | 0.88 | 0.86 | 0.86 |
| 26880x26880        | 1 | 0.98 | 0.89 | 0.90       | 0.89 | 0.88 | 0.87 |

### Hybrid MPI+OpenMp Program with convergence check (with AllReduce calls):

|                       |   |      |      | Timings |       |       |       | 
| --------------------- | - | ---- | ---- | ------- | ----- | ----- | ----- | 
| Grid Size/ Processes - Threads | 1       | 1p-4t   | 2p-8t   | 4p-16t | 9p-36t | 16p-64t | 20p-80t |
| 840x840                        | 0.803   | 0.205   | 0.114   | 0.071  | 0.051  | 0.068   | 0.059   |
| 1680x1680                      | 3.183   | 0.811   | 0.456   | 0.237  | 0.127  | 0.093   | 0.074   |
| 3360x3360                      | 12.700  | 3.208   | 1.776   | 0.938  | 0.434  | 0.264   | 0.242   |
| 6720x6720                      | 50.779  | 12.813  | 7.070   | 3.599  | 1.653  | 0.959   | 0.778   |
| 13440x13440                    | 202.993 | 51.122  | 28.082  | 14.242 | 6.427  | 3.674   | 3.108   |
| 26880x26880                    | 817.426 | 207.031 | 115.019 | 56.769 | 25.465 | 14.352  | 11.619  |

|                    |   |      |      | Speedup |       |       |       |
| ------------------ | - | ---- | ---- | ------- | ----- | ----- | ----- |
| Grid Size/ Threads | 1 | 4    | 8    | 16      | 36    | 64    | 80    |
| 840x840            | 1 | 3.92 | 7.04 | 11.30   | 15.75 | 11.73 | 13.53 |
| 1680x1680          | 1 | 3.93 | 6.98 | 13.40   | 25.05 | 34.27 | 43.05 |
| 3360x3360          | 1 | 3.96 | 7.15 | 13.54   | 29.25 | 48.16 | 52.41 |
| 6720x6720          | 1 | 3.96 | 7.18 | 14.11   | 30.73 | 52.97 | 65.26 |
| 13440x13440        | 1 | 3.97 | 7.23 | 14.25   | 31.59 | 55.25 | 65.32 |
| 26880x26880        | 1 | 3.95 | 7.11 | 14.40   | 32.10 | 56.95 | 70.35 |

|                    |   |      |      | Efficiency |      |      |      |
| ------------------ | - | ---- | ---- | ---------- | ---- | ---- | ---- |
| Grid Size/ Threads | 1 | 4    | 8    | 16         | 36   | 64   | 80   |
| 840x840            | 1 | 0.98 | 0.88 | 0.71       | 0.44 | 0.18 | 0.17 |
| 1680x1680          | 1 | 0.98 | 0.87 | 0.84       | 0.70 | 0.54 | 0.54 |
| 3360x3360          | 1 | 0.99 | 0.89 | 0.85       | 0.81 | 0.75 | 0.66 |
| 6720x6720          | 1 | 0.99 | 0.90 | 0.88       | 0.85 | 0.83 | 0.82 |
| 13440x13440        | 1 | 0.99 | 0.90 | 0.89       | 0.88 | 0.86 | 0.82 |
| 26880x26880        | 1 | 0.99 | 0.89 | 0.90       | 0.89 | 0.89 | 0.88 |

### CUDA Program

| | Timings | |
| ----------- | ----- | ------ |
| Grid Size   | 1 GPU | 2 GPUs |
| 840x840     | 0.011 | 0.007  |
| 1680x1680   | 0.040 | 0.022  |
| 3360x3360   | 0.160 | 0.085  |
| 6720x6720   | 0.569 | 0.317  |
| 13440x13440 | 1.923 | 1.040  |
| 26880x26880 | \-    | 3.748  |

*The big grid could not be computed with 1 GPU because of insufficient VRAM.

## Conclusion: 
As you can see, the CUDA program has the best performance because GPUs have tons of threads (which means they can execute tons of computations in parallel), and extremely fast vram.

## Contributors:
1. [Vasilis Kiriakopoulos](https://github.com/MediaBilly)
2. [Dimitris Koutsakis](https://github.com/koutsd)


