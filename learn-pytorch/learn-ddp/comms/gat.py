import time
import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint


pretty_traceback.install()


def run():
    rank = dist.get_rank()
    v = t.full((3,), rank, dtype=t.float32)
    if rank == 2:
        # This is the gatherer
        vs = [t.empty(3) for _ in range(dist.get_world_size())]
        dprint(f"Sending {v} and gathering...")
        # Blocks until everybody else has sent their data
        dist.gather(v, vs, dst=2)
        dprint(f"Gathered {vs}")
    else:
        # These are the senders
        sleep_secs = 5 + dist.get_rank() * 5
        time.sleep(sleep_secs)
        dprint(f"Sending {v}")
        dist.gather(v, dst=2)
