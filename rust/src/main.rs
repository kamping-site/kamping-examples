use std::ops::Add;

use mpi::{
    datatype::{Partition, PartitionMut},
    traits::*, topology::SimpleCommunicator,
};


fn exclusive_prefix_sum<T: Add<Output=T> + Copy>(init : T, vec: &[T]) -> Vec<T> {
    let mut sum = init;
    let mut out = Vec::with_capacity(vec.len());
    for i in 0..vec.len() {
        out.push(sum);
        sum = sum + vec[i];
    }
    out
}

fn bsp_exchange(comm: &SimpleCommunicator, data : &Vec<i32>, scounts : &Vec<i32>) -> Vec<i32> {
    let sdispls = exclusive_prefix_sum(0, &scounts);
    let mut rcounts = vec![0 as i32; comm.size() as usize];
    comm.all_to_all_into(scounts, &mut rcounts);
    let rdispls = exclusive_prefix_sum(0, &rcounts);
    let sendbuf = Partition::new(data, scounts.clone(), sdispls);
    let recv_size = (rcounts.last().unwrap() + rdispls.last().unwrap()) as usize;
    let mut recv_data = vec![0; recv_size];
    let mut recvbuf = PartitionMut::new(&mut recv_data, rcounts, rdispls);
    comm.all_to_all_varcount_into(&sendbuf, &mut recvbuf);
    recv_data
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let mut data = Vec::new();
    for i in 0..comm.size() {
        for _ in 0..i {
            data.push(i);
        }
    }
    let scounts: Vec<i32> = (0..comm.size()).collect();
    bsp_exchange(&comm, &data, &scounts);
}
