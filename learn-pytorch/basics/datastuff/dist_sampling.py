import os
import pickle
from glob import glob

import click
import numpy as np
import pretty_traceback
import torch as t
import torch.distributed as dist
import torch.utils.data as td

pretty_traceback.install()

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29500


class Simple(td.Dataset):
    def __init__(self, m=100, n=3):
        self._data = np.array([np.full(n, i + 1) for i in range(m)])

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        return self._data[idx]


def main(rank, world, nrows):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = str(MASTER_PORT)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["RANK"] = str(rank)
    dist.init_process_group("gloo")

    ds = Simple(m=nrows)
    sampler = td.DistributedSampler(ds)
    dl = td.DataLoader(ds, batch_size=8, sampler=sampler)
    data = []
    for batch in dl:
        for x in batch:
            data.append(x[0].item())

    with open(f"data_{rank}.pkl", "wb") as f:
        pickle.dump(data, f)

    dist.destroy_process_group()


@click.command()
@click.option("--nrows", default=100, help="Number of rows in the dataset.")
@click.option("--world", default=2, help="Number of processes to start.")
def launcher_main(world, nrows):
    t.multiprocessing.spawn(main, args=(world, nrows), nprocs=world, join=True)

    input("All workers have exited, press ENTER to start analysis: ")
    all_data = []
    for rank in range(world):
        with open(f"data_{rank}.pkl", "rb") as f:
            all_data.append(pickle.load(f))

    marked = [False] * (nrows + 1)
    dups = []
    for rank, data in enumerate(all_data):
        print(f"Processing data from rank: {rank}")
        for x in data:
            if marked[x]:
                dups.append(x)
            else:
                marked[x] = True
        marked_counts = len(list(filter(lambda x: x, marked)))
        print(f"Count of marked: {marked_counts}")

    unmarked = [idx for idx in range(len(marked)) if not marked[idx]]
    print("Unmarked: ", unmarked)
    print("Duplicates: ", dups)

    for pklfile in glob("./*.pkl"):
        os.remove(pklfile)


if __name__ == "__main__":
    launcher_main()
