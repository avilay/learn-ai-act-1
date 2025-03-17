import os
from datetime import datetime

import numpy as np
from cprint import cprint
from torch.utils.data import DataLoader, Dataset, get_worker_info

rng = np.random.default_rng()


def log_worker(msg):
    wi = get_worker_info()
    pid = os.getpid()
    if wi is not None:
        wid = wi.id
        n_workers = wi.num_workers
    else:
        wid, n_workers = -1, -1
    now = datetime.now().strftime("%H:%M:%S")
    color = wid + 1
    cprint(color, f"{now} - ({pid}){wid}/{n_workers}: {msg}")


class MyMappedDataset(Dataset):
    def __init__(self, n=5, m=10):
        log_worker("Instantiating dataset")
        self._m = m
        self._n = n

    def __getitem__(self, idx):
        log_worker(f"Fetching data[{idx}]")
        return np.full((self._n,), fill_value=idx), rng.choice([0.0, 1.0])

    def __len__(self):
        return self._m


def main():
    log_worker("Starting main")
    ds = MyMappedDataset(m=24)
    dl = DataLoader(ds, shuffle=False, num_workers=2, batch_size=3, prefetch_factor=1)
    for i, (X, y) in enumerate(dl):
        input("Press [ENTER] to accept batch:\n")
        print(X, y)
    print("Done")


if __name__ == "__main__":
    main()
