import os

import click
import pretty_traceback
import torch as t
import torch.distributed as dist

import bcastol
import p2p
import allred
import bcast
import allgat
import mean
import scat
import gat
import red
import redscat
import all2all

pretty_traceback.install()


MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29500


def main(rank, world, task):
    tasks = {
        "p2p": (p2p.run_master, p2p.run_worker),
        "allred": (allred.run, allred.run),
        "bcastol": (bcastol.run_master, bcastol.run_worker),
        "bcast": (bcast.run_master, bcast.run_worker),
        "allgat": (allgat.run, allgat.run),
        "mean": (mean.run, mean.run),
        "scat": (scat.run_master, scat.run_worker),
        "gat": (gat.run, gat.run),
        "red": (red.run, red.run),
        "redscat": (redscat.run, redscat.run),
        "all2all": (all2all.run, all2all.run),
    }
    run_master, run_worker = tasks[task]

    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["RANK"] = str(rank)
    dist.init_process_group("gloo")

    if rank == 0:
        run_master()
    else:
        run_worker()


@click.command()
@click.option(
    "--bcastol", "task", flag_value="bcastol", help="demo for broadcast_object_list."
)
@click.option("--allred", "task", flag_value="allred", help="demo for all_reduce.")
@click.option("--red", "task", flag_value="red", help="demo for reduce.")
@click.option("--p2p", "task", flag_value="p2p", help="demo for send and recv.")
@click.option("--bcast", "task", flag_value="bcast", help="demo for broadcast.")
@click.option("--allgat", "task", flag_value="allgat", help="demo for all_gather.")
@click.option("--gat", "task", flag_value="gat", help="demo for gather.")
@click.option(
    "--mean", "task", flag_value="mean", help="demo of how to calculate mean."
)
@click.option("--scat", "task", flag_value="scat", help="demo how to scatter.")
@click.option(
    "--redscat", "task", flag_value="redscat", help="demo how to reduce-scatter."
)
@click.option("--all2all", "task", flag_value="all2all", help="demo of all_to_all.")
@click.option(
    "--world", default=2, help="World size, i.e., the number of processes to launch."
)
def launcher_main(world, task):
    t.multiprocessing.spawn(main, args=(world, task), nprocs=world, join=True)


if __name__ == "__main__":
    launcher_main()
