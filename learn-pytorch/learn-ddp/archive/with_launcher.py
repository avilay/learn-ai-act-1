"""
$ alias torchlaunch="python /Users/avilay/opt/anaconda3/envs/ai/lib/python3.7/site-packages/torch/distributed/launch.py"

To start 4 processes on a single node:
$ torchlaunch --nnode=1 --node_rank=0 --nproc_per_node=4 with_launcher.py --local_world_size=4 --cookie="Oatmeal Raisin"

To start 2 processes per node on two nodes:
Step 1: Pull the pytorch image from Docker hub
$ docker pull pytorch/pytorch

Step 2: Start two Docker containers - call one master and the other trainer
The -dit flag with start the container's default ENTRYPOINT/CMD (/bin/bash in case of PyTorch) in detached but interactive terminal.
$ docker run -dit --name master pytorch/pytorch
$ docker run -dit --name trainer pytorch/pytorch

Step 3: Get the IP addresses of both the containers
$ docker network inspect bridge
Lets assume master IP is 172.17.0.2 and trainer IP is 172.17.0.3

Step 4: Copy this file to both the containers
$ docker container cp ./with_launcher.py master:/workspace/with_launcher.py
$ docker container cp ./with_launcher.py trainer:/workspace/with_launcher.py

Step 5: Run launcher (remember to set the alias) on the trainer first
$ docker attach trainer
# torchlaunch --nproc_per_node=2 \
              --nnodes=2 \
              --node_rank=1 \
              --master_addr="172.17.0.2" \
              --master_port=8888
              with_launcher.py --local_world_size=2 --cookie="Snicker Doodle"

Step 6: Run launcher (remember to set the alias) on the master
$ docker attach master
# torchlaunch --nproc_per_node=2 \
              --nnodes=2 \
              --node_rank=0 \
              --master_addr="172.17.0.2" \
              --master_port=8888 \
              with_launcher.py --local_world_size=2 --cookie="Oatmeal Raisin"
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


@click.command()
@click.option("--local_rank", type=int, default=0, help="This is requried by the launcher.")
@click.option("--local_world_size", type=int, default=1, help="This is what my program needs.")
@click.option(
    "--cookie", type=str, default="Chocolate Chip", help="More args that my program needs."
)
def main(local_rank, local_world_size, cookie):
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    global_rank = os.environ["RANK"]
    global_world_size = os.environ["WORLD_SIZE"]
    print(f"Process {os.getpid()}:")
    print(f"\tMaster={master_addr}:{master_port}")
    print(f"\tLocal World={local_rank}/{local_world_size}")
    print(f"\tGlobal World={global_rank}/{global_world_size}")

    dist.init_process_group("gloo", rank=local_rank, world_size=local_world_size)
    print(f"\tDist={dist.get_rank()}/{dist.get_world_size()}::{dist.get_backend()}")
    train(cookie)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
