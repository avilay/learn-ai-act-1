import time

import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run_master():
    v = t.tensor([1.0, 2.0, 3.0])
    dprint(f"Before broadcast: {v}")
    # Blocks untill some (most?) of the workers have accepted this broadcast
    dist.broadcast(v, src=0)
    dprint(f"After broadcast: {v}")


def run_worker():
    v = t.tensor([0.0, 0.0, 0.0])
    sleep_secs = 5 + dist.get_rank() * 5
    dprint(f"Before broadcast: {v}, now will sleep for {sleep_secs} seconds")
    time.sleep(sleep_secs)
    dist.broadcast(v, src=0)
    dprint(f"After broadcast: {v}")
