import os
from pathlib import Path

import click
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as td
import torchvision as tv

# hparams as consts for now
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.5
N_EPOCHS = 10


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout(0.25)
        self.dropout2 = t.nn.Dropout(0.5)
        self.fc1 = t.nn.Linear(9216, 128)
        self.fc2 = t.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = t.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits


def my_dataset():
    n_partitions = dist.get_world_size()
    rank = dist.get_rank()
    xforms = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307), (0.3081))]
    )
    dataroot = Path.home() / "mldata" / "mnist"
    dataset = tv.datasets.MNIST(dataroot, train=True, download=True, transform=xforms)
    partition_len = len(dataset) // n_partitions
    final_partition_len = len(dataset) - (partition_len) * (n_partitions - 1)
    partition_lens = [partition_len] * (n_partitions - 1) + [final_partition_len]
    dataset_splits = td.random_split(dataset, partition_lens)
    return dataset_splits[rank]


def my_batch_size():
    n_partitions = dist.get_world_size()
    rank = dist.get_rank()
    batch_size = BATCH_SIZE // n_partitions
    final_batch_size = BATCH_SIZE - batch_size * (n_partitions - 1)
    batch_sizes = [batch_size] * (n_partitions - 1) + [final_batch_size]
    return batch_sizes[rank]


def run(rank, n_partitions):
    trainset = my_dataset()
    batch_size = my_batch_size()
    traindl = td.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = Net()
    optim = t.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = t.nn.CrossEntropyLoss()

    for epoch in range(N_EPOCHS):
        epoch_losses = []
        for x, y in traindl:
            optim.zero_grad()
            logits = model(x)
            loss = loss_fn(logits)
            epoch_losses.append(loss.item())
            loss.backward()
            avergae_gradients(model)
            optim.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Rank {rank}: epoch = {epoch} loss = {avg_loss:.3f}")


def avergae_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size()


def init_process(rank, n_partitions):
    print(f"init_process: rank={rank} n_partitions={n_partitions}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=n_partitions)
    run(rank, n_partitions)


@click.command()
@click.option("--nparts", default=2, help="Number of partitions")
def main(nparts):
    n_partitions = nparts
    mp.set_start_method("spawn")

    print("Starting processes")
    procs = []
    for rank in range(n_partitions):
        proc = mp.Process(target=init_process, args=(rank, n_partitions, run))
        proc.start()
        procs.append(proc)

    print("Waiting for processes to terminiate")
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
