"""
This program demos how to calculate the mean of all the elements in two tensors
living in different ranks. The tensors may be of different lengths.
"""
import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def get_all_lens(mylen):
    world_sz = dist.get_world_size()
    all_lens = [t.empty(1, dtype=t.int8) for _ in range(world_sz)]
    dist.all_gather(all_lens, t.tensor([mylen], dtype=t.int8))
    return [all_lens[i].item() for i in range(len(all_lens))]


def run():
    myrank = dist.get_rank()
    mylen = t.randint(2, 5, (1,)).item()
    v = t.randn(mylen)
    dprint(v)
    all_lens = get_all_lens(mylen)

    world_sz = dist.get_world_size()
    vs = [None] * world_sz
    for rank in range(world_sz):
        if rank == myrank:
            # broadcast my tensor to everybody
            dist.broadcast(v, src=myrank)
            vs[rank] = v  # just copy my tensor here
        else:
            # get tensor from this rank
            u = t.empty(all_lens[rank])
            dist.broadcast(u, src=rank)
            vs[rank] = u
    mean = t.cat(vs).mean().item()
    dprint(f"My mean is {mean}")
