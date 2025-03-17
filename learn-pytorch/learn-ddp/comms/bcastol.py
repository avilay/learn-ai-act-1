import time

import pretty_traceback
import torch.distributed as dist
from distml import dprint
import numpy as np

pretty_traceback.install()


def run_master():
    v1 = np.array([1.0, 2.0, 3.0]).astype(np.float32)
    dprint(f"Before broadcast: {v1}")
    # Blocks untill some (most?) of the workers have accepted this broadcast
    dist.broadcast_object_list([v1])
    dprint(f"After broadcast: {v1}")


def run_worker():
    objs = [None]
    sleep_secs = 5 + dist.get_rank() * 5
    dprint(f"Now sleeping for {sleep_secs} seconds.")
    time.sleep(sleep_secs)
    dist.broadcast_object_list(objs)
    dprint(f"After broadcast: {objs}")
