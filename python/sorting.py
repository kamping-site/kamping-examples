from mpi4py import MPI
import numpy as np
import sys
import math

def main():
    # data = np.random.randint(0, 10, n)
    data = []
    for i in range(MPI.COMM_WORLD.size):
        data += [i] * i
    comm = MPI.COMM_WORLD
    data = np.array(data)
    counts = np.arange(MPI.COMM_WORLD.size)
    print(data)
    rbuf = np.empty(comm.rank * comm.size, dtype=np.int64)
    comm.Alltoallv([data, counts], rbuf)
    print(rbuf)

if __name__ == "__main__":
    main()
