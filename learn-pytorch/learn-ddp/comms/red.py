import time
import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run():
    rank = dist.get_rank()
    if rank == 2:
        # This is receiver of the reduced tensor
        v = t.empty(3)
        dprint("Reducing...")
        dist.reduce(v, dst=2, op=dist.ReduceOp.SUM)
        dprint(f"Reduced {v}")
    else:
        # These are the senders
        v = t.full((3,), rank, dtype=t.float32)
        sleep_secs = 5 + rank * 2
        time.sleep(sleep_secs)
        dprint(f"Sending {v}")
        dist.reduce(v, dst=2, op=dist.ReduceOp.SUM)
