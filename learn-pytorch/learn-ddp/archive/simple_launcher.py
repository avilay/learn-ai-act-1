import hydra
import torch as t
from termcolor import cprint
from colorama import init
import random
import time

init()

rank_color = ["yellow", "green", "blue", "magenta", "cyan", "red", "grey", "white"]


def local_main(local_rank, cookie):
    # local_rank is automatically supplied by the spawn method.
    # cookie is part of the args param to spawn method.
    sleep_time = random.randint(1, 7)
    cprint(
        f"{local_rank}: Munching {cookie} for {sleep_time} seconds",
        rank_color[local_rank],
    )
    time.sleep(sleep_time)
    cprint(f"{local_rank}: Done", rank_color[local_rank])


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    # join=False does not really work, wonder what its use case is
    t.multiprocessing.spawn(
        local_main, args=(cfg.cookie,), nprocs=cfg.nprocs, join=True
    )
    print("Launcher exiting.")


if __name__ == "__main__":
    main()
