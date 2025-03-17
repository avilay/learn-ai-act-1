import os

import hydra
import torch as t
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import numpy as np

from rank_printer import dist_print


class ToyModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(10, 10)
        self.fc2 = t.nn.Linear(10, 5)

    def forward(self, batch_x):
        x = F.relu(self.fc1(batch_x))
        batch_y_hat = self.fc2(x)
        return batch_y_hat


def avg(model):
    tot = 0
    num = 0
    for param in model.parameters():
        tot += param.sum().item()
        num += np.prod(list(param.shape))
    avg = tot / num
    return avg


def train():
    model = ToyModel()
    ddp_model = DDP(model)
    loss_fn = t.nn.MSELoss()
    optim = t.optim.SGD(ddp_model.parameters(), lr=0.01)

    dist_print(f"Before train param avg: {avg(model):.4f}")

    for _ in range(50):
        inputs = t.randn(20, 10)
        targets = t.randn(20, 5)
        optim.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optim.step()

    dist_print(f"After train param avg: {avg(model):.4f}")


def main(rank, master_addr, master_port, world_size):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group("gloo")
    train()
    dist.destroy_process_group()


@hydra.main(config_path=".", config_name="config")
def launcher_main(cfg):
    master_addr = cfg.job.master_addr
    master_port = cfg.job.master_port
    world_size = cfg.job.world_size

    if cfg.job.local:
        # No need to pass the rank explicitly to main, it is done automatically by spawn
        t.multiprocessing.spawn(
            main,
            args=(master_addr, master_port, world_size),
            nprocs=world_size,
            join=True,
        )
    else:
        rank = cfg.job.rank
        main(rank, master_addr, master_port, world_size)


if __name__ == "__main__":
    launcher_main()
