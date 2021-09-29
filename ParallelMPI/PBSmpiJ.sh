#!/bin/bash

# JobName #
#PBS -N JparmpiJacobi

#Which Queue to use #
#PBS -q N10C80

# Max Wall time, Example 1 Minute #
#PBS -l walltime=00:20:00

# How many nodes and tasks per node
#PBS -l select=7:ncpus=8:mpiprocs=7:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun -np 49 jacobi_parallel_mpi.x < input
