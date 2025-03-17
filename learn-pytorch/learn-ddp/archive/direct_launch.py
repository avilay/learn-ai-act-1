"""
$ MASTER_ADDR=127.0.0.1 MASTER_PORT=10000 python direct_launch.py nprocs=2 cookie="Snicker Doodle"
"""

import click
import os
import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


class ToyModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(10, 10)
        self.fc2 = t.nn.Linear(10, 5)

    def forward(self, batch_x):
        x = F.relu(self.fc1(batch_x))
        batch_y_hat = self.fc2(x)
        return batch_y_hat


def train(cookie):
    print(f"training with {cookie}")
    model = ToyModel()
    ddp_model = DDP(model)
    loss_fn = t.nn.MSELoss()
    optim = t.optim.SGD(ddp_model.parameters(), lr=0.01)

    inputs = t.randn(20, 10)
    targets = t.randn(20, 5)

    optim.zero_grad()
    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optim.step()


# First param is always the local rank which is provided by the spawn function
# rest of the args are provided in the args tuple
def main(local_rank, local_world_size, cookie):
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    print(f"Process {os.getpid()}:")
    print(f"\tMaster={master_addr}:{master_port}")
    print(f"\tWorld={local_rank}/{local_world_size}")

    dist.init_process_group("gloo", rank=local_rank, world_size=local_world_size)
    print(f"\tDist={dist.get_rank()}/{dist.get_world_size()}::{dist.get_backend()}")
    train(cookie)
    dist.destroy_process_group()


@click.command()
@click.option("--nprocs", default=1, help="Number of processes to launch.")
@click.option("--cookie", default="Chocolate Chip", help="Additional args to the trainer")
def my_launcher(nprocs, cookie):
    t.multiprocessing.spawn(main, args=(nprocs, cookie), nprocs=nprocs, join=True)


if __name__ == "__main__":
    my_launcher()
