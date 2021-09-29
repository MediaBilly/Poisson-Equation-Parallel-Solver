all:
	mpicc -O3 -lm -o jacobi_parallel_mpi.x jacobi_parallel_mpi.c

mpiP:
# module load mpip
	mpicc -O3 -g -L/opt/mpiP-3.5/lib -lmpiP -lbfd -lunwind -lm -o jacobi_parallel_mpi.x jacobi_parallel_mpi.c