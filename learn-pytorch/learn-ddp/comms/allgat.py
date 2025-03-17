import time

import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run():
    rank, world_size = dist.get_rank(), dist.get_world_size()
    vs = [t.empty(3, dtype=t.float32) for _ in range(world_size)]
    v = t.full((3,), rank, dtype=t.float32)
    dprint(f"Sending {v} and gathering...")
    dist.all_gather(vs, v)
    dprint(f"Gathered {vs}")
