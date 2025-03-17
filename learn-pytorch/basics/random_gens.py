"""
This is the pytorch tensor equivalent of the //learn-scipy/numpy/random_gens.py to showcase
how random seeds work.
"""

from multiprocessing import Process

import click
import torch as t
from cprint import cprint


def generate(rank, seed):
    t.random.manual_seed(seed)
    fc = t.nn.Linear(5, 1)
    for param in fc.parameters():
        cprint(rank, param)


@click.command()
@click.option("--seed", type=int, required=True)
@click.option("--nprocs", default=5)
def main(seed, nprocs):
    procs = [Process(target=generate, args=(i, seed)) for i in range(nprocs)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
