import time
import pretty_traceback
import torch as t
import torch.distributed as dist
from distml import dprint

pretty_traceback.install()


def run_master():
    return run_worker()


def run_worker():
    to_send = [t.rand(3) for _ in range(dist.get_world_size())]
    recvd = t.empty(3)
    if (rank := dist.get_rank()) == 2:
        dprint(f"Scattering tensors {to_send}")
        dist.scatter(recvd, to_send, src=rank)
    else:
        sleep_secs = 5 + dist.get_rank() * 5
        time.sleep(sleep_secs)
        dist.scatter(recvd, src=2)
    dprint(f"Received tensor {recvd}")
