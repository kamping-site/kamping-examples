from mpi4py import MPI
import numpy as np
import sys
import math
import argparse

def globally_sorted(comm: MPI.Comm, data, original_data):
    global_data = comm.gather(data)
    if global_data is None:
        global_data = [[]]
    global_data_original = comm.gather(original_data)
    if global_data_original is None:
        global_data_original = [[]]
    global_data = np.concatenate(global_data)
    global_data_original = np.concatenate(global_data_original)
    global_data.sort()
    global_data_original.sort()
    return (global_data == global_data_original).all()

def generate_data(n_local, seed):
    gen = np.random.MT19937(seed + MPI.COMM_WORLD.rank)
    max_int = np.iinfo(np.uint64).max
    return np.random.Generator(gen).integers(0, high=max_int, size=n_local, dtype=np.uint64)

def sort(comm: MPI.Comm, data, seed):
    oversampling_ratio = int(16 * math.log2(comm.size)) + 1;
    gen = np.random.Generator(np.random.MT19937(seed))
    local_samples = gen.choice(data, size=oversampling_ratio)
    global_samples = np.empty(local_samples.size * comm.size, dtype=local_samples.dtype)
    comm.Allgather(local_samples, global_samples)
    global_samples.sort()
    global_samples = global_samples[oversampling_ratio::oversampling_ratio]
    buckets = np.empty(comm.size, dtype=object)
    for i in range(len(buckets)):
        buckets[i] = []
    bucket_id = np.digitize(data, global_samples)
    for i in range(len(data)):
        if buckets[bucket_id[i]] is None:
            buckets[bucket_id[i]] = []
        buckets[bucket_id[i]].append(data[i])
    data = np.concatenate(buckets)
    scounts = np.array([len(b) for b in buckets], dtype=np.intc)
    rcounts = np.empty(comm.size, dtype=np.intc)
    comm.Alltoall(scounts, rcounts)
    rbuf = np.empty(rcounts.sum(), dtype=data.dtype)

    comm.Alltoallv([data, scounts], [rbuf, rcounts])
    rbuf.sort()
    data = rbuf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_local", type=int, required=True)
    args = parser.parse_args()
    data = generate_data(args.n_local, args.seed)
    original_data = data.copy()
    local_seed = args.seed + MPI.COMM_WORLD.rank + MPI.COMM_WORLD.size
    sort(MPI.COMM_WORLD, data, local_seed)
    correct = globally_sorted(MPI.COMM_WORLD, data, original_data)
    if MPI.COMM_WORLD.rank == 0:
        print(f"correct: {correct}")

if __name__ == "__main__":
    main()
