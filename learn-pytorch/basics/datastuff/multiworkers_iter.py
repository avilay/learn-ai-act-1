import os
import random
import time
from datetime import datetime

import click
import numpy as np
import torch.utils.data as td
from cprint import cprint


def log_worker(txt):
    wid = td.get_worker_info().id if td.get_worker_info() is not None else 999  # type: ignore
    pid = os.getpid()
    now = datetime.now().strftime("%H:%M:%S")
    cprint(wid, f"{now} - ({pid})w{wid}: {txt}")


class MySlowBatchedDataset(td.IterableDataset):
    def __init__(self, m=10, n=3, slow=True, batch_size=2, start=1):
        super().__init__()
        log_worker("Instantiating dataset")
        self.m = 10
        self._n = 3
        self._slow = slow
        self._batch_size = batch_size
        self.start = start

    def __iter__(self):
        for i in range(self.m):
            x = np.full((self._batch_size, self._n), fill_value=i + self.start)
            y = np.random.choice([0, 1], size=self._batch_size, p=[0.7, 0.3])
            if self._slow:
                log_worker(f"Starting to process row [{i}]")
                time.sleep(random.randint(1, 5))
            log_worker(f"Returning row [{i}]")
            yield x, y

    def __len__(self):
        return self.m


def worker_init_fn(wid):
    wi = td.get_worker_info()
    ds = wi.dataset  # type: ignore
    ds.start = wid * 1000  # type: ignore
    log_worker(f"Initialized worker ds to start with {ds.start}")  # type: ignore


@click.command()
@click.option("--workers", default=0, help="The number of dataloader workers")
@click.option("--slow/--fast", default=True)
@click.option("--prefetch", default=2, help="Set the prefetch_factor")
@click.option("--batch", default=1, help="The batch size")
@click.option("--size", default=10, help="Number of rows in the dataset")
def main(workers, slow, prefetch, batch, size):
    ds = MySlowBatchedDataset(slow=slow, m=size, batch_size=batch)
    if workers > 0:
        dl = td.DataLoader(
            ds,
            shuffle=False,
            num_workers=workers,
            prefetch_factor=prefetch,
            batch_size=None,
            worker_init_fn=worker_init_fn,
        )
    else:
        dl = td.DataLoader(ds, batch_size=None)
    try:
        for x, y in dl:
            if slow:
                time.sleep(10)
            now = datetime.now().strftime("%H:%M:%S")
            print(f"{now}\n{x}, {y}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
