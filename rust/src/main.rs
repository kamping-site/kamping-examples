use clap::Parser;
use mpi::Count;
use rand::prelude::*;
use rand::Fill;
use rand_pcg::Pcg64Mcg;
use std::ops::Add;

use mpi::{
    datatype::{Partition, PartitionMut},
    topology::SimpleCommunicator,
    traits::*,
};

fn generate_data<T>(n_local: usize, seed: u64) -> Vec<T>
where
    T: Default + Clone,
    [T]: Fill,
{
    let mut gen = Pcg64Mcg::seed_from_u64(seed);
    let mut data = vec![T::default(); n_local];
    gen.fill(data.as_mut_slice());
    data
}

fn exclusive_prefix_sum<T: Add<Output = T> + Copy>(init: T, vec: &[T]) -> Vec<T> {
    let mut sum = init;
    let mut out = Vec::with_capacity(vec.len());
    for i in 0..vec.len() {
        out.push(sum);
        sum = sum + vec[i];
    }
    out
}

fn gather_on_root<T>(comm: &SimpleCommunicator, v: &Vec<T>) -> Option<Vec<T>>
where
    T: Default + Clone + Equivalence,
{
    let root = comm.process_at_rank(0);
    if comm.rank() == 0 {
        let mut rcounts = vec![0 as Count; comm.size() as usize];
        root.gather_into_root(&(v.len() as Count), &mut rcounts);
        let rdispls: Vec<Count> = rcounts
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();
        let mut buf = vec![T::default(); rcounts.iter().sum::<Count>() as usize];
        let mut partition = PartitionMut::new(&mut buf, rcounts, rdispls);
        root.gather_varcount_into_root(v, &mut partition);
        Some(buf)
    } else {
        root.gather_into(&(v.len() as Count));
        root.gather_varcount_into(v);
        None
    }
}

fn globally_sorted<T>(comm: &SimpleCommunicator, data: &Vec<T>, original_data: &Vec<T>) -> bool
where
    T: Equivalence + Default + Clone + PartialEq + Ord,
{
    let global_data = gather_on_root(comm, data).iter_mut().for_each(|x| x.sort());
    let global_original_data = gather_on_root(comm, original_data)
        .iter_mut()
        .for_each(|x| x.sort());
    return global_data == global_original_data;
}

fn sort<T>(comm: &SimpleCommunicator, data: &mut Vec<T>, seed: u64)
where
    T: Equivalence + Eq + Ord + Default + Copy,
{
    let oversampling_ratio = 16 * comm.size().ilog2() + 1;
    let mut gen = Pcg64Mcg::seed_from_u64(seed);
    let local_samples = (0..oversampling_ratio)
        .map(|_| {
            let idx = gen.gen_range(0..data.len());
            data[idx]
        })
        .collect::<Vec<T>>();
    let mut global_samples = vec![T::default(); local_samples.len() * (comm.size() as usize)];
    comm.all_gather_into(&local_samples, &mut global_samples);
    global_samples.sort();
    let global_samples = (0..comm.size() as usize - 1)
        .map(|i| global_samples[(i + 1) * oversampling_ratio as usize])
        .collect::<Vec<T>>();
    let mut buckets = vec![Vec::<T>::new(); comm.size() as usize];
    for elem in data.iter() {
        let bucket_idx = global_samples.binary_search(&elem).unwrap_or_else(|x| x);
        buckets[bucket_idx].push(*elem);
    }
    data.clear();
    let scounts = buckets
        .iter()
        .map(|x| x.len() as Count)
        .collect::<Vec<Count>>();
    let mut rcounts = vec![0 as Count; comm.size() as usize];
    comm.all_to_all_into(&scounts, &mut rcounts);
    let sdispls = exclusive_prefix_sum(0, &scounts);
    let rdispls = exclusive_prefix_sum(0, &rcounts);
    let sbuf = buckets.into_iter().flatten().collect::<Vec<T>>();
    let sbuf = Partition::new(&sbuf, scounts, sdispls);
    data.resize(
        (rcounts.last().unwrap() + rdispls.last().unwrap()) as usize,
        T::default(),
    );
    let mut rbuf = PartitionMut::new(data, rcounts, rdispls);
    comm.all_to_all_varcount_into(&sbuf, &mut rbuf);
    data.sort();
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    n_local: usize,
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let args = Args::parse();
    let comm = universe.world();
    let mut data = generate_data::<u64>(args.n_local, args.seed + comm.rank() as u64);
    let original_data = data.clone();
    let local_seed = args.seed + comm.rank() as u64 + comm.size() as u64;
    sort(&comm, &mut data, local_seed);
    let result = globally_sorted(&comm, &data, &original_data);
    if comm.rank() == 0 {
        println!("correct: {}", result);
    }
}
