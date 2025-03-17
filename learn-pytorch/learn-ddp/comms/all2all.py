import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint


pretty_traceback.install()


def run():
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # For a world with 3 workers vs will for rank 0 will be -
    # [[1, 1, 1],
    #  [10, 10, 10],
    #  [100, 100, 100]]
    vs = [
        t.full((3,), (rank + 1) * (10**i), dtype=t.float32) for i in range(world_size)
    ]
    us = [t.empty(3) for _ in range(world_size)]
    dprint(f"Before all_to_all: {us}")
    dist.all_to_all(us, vs)
    dprint(f"After all_to_all: {us}")
