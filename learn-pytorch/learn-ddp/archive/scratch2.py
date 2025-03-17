from multiprocessing import process
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys


def run(rank, size):
    """ Distributed function to be implemented later. """
    pass


def init_process(rank, size, fn, backend="gloo"):
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    print("Starting init_process_group")
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    print("Exiting")
    # dist.destroy_process_group()


if __name__ == "__main__":
    # processes = []
    # mp.set_start_method("spawn")
    # size = 2
    # for rank in range(size):
    #     p = mp.Process(target=init_process, args=(rank, size, run))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
    init_process(int(sys.argv[1]), 2, run)
