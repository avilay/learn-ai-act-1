import os
import random
import time

import hydra
import torch as t
import torch.distributed as dist

from rank_printer import dist_print


def main(rank, master_addr, master_port, world_size):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist_print("Starting init_process_group..")
    dist.init_process_group("nccl")

    DEVICE = t.device(f"cuda:{rank}")

    with t.cuda.device(DEVICE):
        sleep_time = random.randint(3, 10)
        dist_print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)
        dist_print("Exiting.")

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
