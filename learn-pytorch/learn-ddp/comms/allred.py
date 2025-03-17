"""
Comment one of the dist.reduce calls in the both the run_master and run_worker functions.

dist.reduce(v1, 1) is saying to reduce the tensor v1 on the rank:1 process.
dist.reduce(v2, 1) is saying to reduce the tensor v2 on the rank:1 process.

This will result in v1 being unchanged on rank:0 but v2 will be overwritten with the new reduced values on rank:1
"""

import time

import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run():
    v = t.rand(3)
    dprint(f"Before reducing: {v}")
    dist.all_reduce(v)
    dprint(f"After reducing: {v}")
