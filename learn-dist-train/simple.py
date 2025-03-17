#!/usr/bin/env python
import os
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import click
import sys


def simple(rank, n_partitions):
    print(f"run: rank={rank} n_partitions={n_partitions}")
    tensor = t.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print(f"Rank {rank}: tensor[0] = {tensor[0]}")


def allred(rank, n_partitions):
    # group = dist.new_group([0, 1])
    tensor = t.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank {rank}: data = {tensor[0]}")


def allgather(rank, n_partitions):
    tensor = t.Tensor([rank])
    tensor_list = [t.zeros(1) for _ in range(n_partitions)]
    dist.all_gather(tensor_list=tensor_list, tensor=tensor)
    print(f"Rank {rank}: Out = {tensor_list}")


def init_process(rank, n_partitions, fn):
    print(f"init_process: rank={rank} n_partitions={n_partitions}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=n_partitions)
    fn(rank, n_partitions)


@click.command()
@click.argument("funcname")
@click.option("--nparts", default=2, help="Number of partitions")
def main(funcname, nparts):
    func = getattr(sys.modules[__name__], funcname)

    n_partitions = nparts
    mp.set_start_method("spawn")

    print("Starting processes")
    procs = []
    for rank in range(n_partitions):
        proc = mp.Process(target=init_process, args=(rank, n_partitions, func))
        proc.start()
        procs.append(proc)

    print("Waiting for processes to terminiate")
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
