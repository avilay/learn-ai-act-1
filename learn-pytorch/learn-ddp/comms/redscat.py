from re import T
import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run():
    output = t.empty(3)
    world_size = dist.get_rank(), dist.get_world_size()
    input_list = [t.rand(world_size * 3) for _ in range(world_size)]
    dist.reduce_scatter(output, input_list)
    dprint(output)
