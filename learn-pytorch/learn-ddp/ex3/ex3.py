import os

import hydra
import torch as t
import torch.distributed as dist

from rank_printer import dist_print

import time
import random


def run_master():
    # tensor = t.Tensor([1.0])
    # dist.broadcast(tensor, src=dist.get_rank())
    # dist.send(tensor=tensor, dst=1)
    # dist.all_reduce(tensor)
    # dist_print(f"Have data: {tensor[0]}")
    sleep_time = random.randint(1, 7)
    dist_print(f"Sleeping for {sleep_time} seconds.")
    time.sleep(sleep_time)
    dist_print("Hit barrier.")
    dist.barrier()
    dist_print("Got past barrier.")


def run_trainer():
    # tensor = t.Tensor([2.0])
    # dist.recv(tensor=tensor, src=0)
    # dist.all_reduce(tensor)
    # dist.broadcast(tensor, src=0)
    # dist_print(f"Have data: {tensor[0]}")
    sleep_time = random.randint(1, 7)
    dist_print(f"Sleeping for {sleep_time} seconds.")
    time.sleep(sleep_time)
    dist_print("Hit barrier.")
    dist.barrier()
    dist_print("Got past barrier.")


def main(rank, master_addr, master_port, world_size):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo")
    dist_print("Process group initialized.")

    if rank == 0:
        run_master()
    else:
        run_trainer()

    dist.destroy_process_group()


@hydra.main(config_path=".", config_name="config")
def launcher_main(cfg):
    master_addr = cfg.job.master_addr
    master_port = cfg.job.master_port
    world_size = cfg.job.world_size

    if cfg.job.local:
        # No need to pass the rank explicitly to main, it is done automatically by spawn
        t.multiprocessing.spawn(
            main,
            args=(master_addr, master_port, world_size),
            nprocs=world_size,
            join=True,
        )
    else:
        rank = cfg.job.rank
        main(rank, master_addr, master_port, world_size)


if __name__ == "__main__":
    launcher_main()
