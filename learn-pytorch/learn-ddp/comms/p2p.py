import time

import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run_master():
    v1 = t.rand(5)
    dist.send(v1, dst=1)
    sleep_secs = t.randint(2, 5, (1,)).item()
    dprint(f"Vector {v1} sent to worker. Now sleeping for {sleep_secs} seconds.")
    time.sleep(sleep_secs)


def run_worker():
    v2 = t.empty(5)
    sleep_secs = t.randint(2, 5, (1,)).item()
    dprint(f"Sleeping for {sleep_secs} seconds.")
    time.sleep(sleep_secs)
    dist.recv(v2, src=0)
    dprint(f"Rcvd vector {v2}")
