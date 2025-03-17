"""
To train -
```
python dist_binclass.py --train --rank=0
python dist_binclass.py --train --rank=1
```

To test -
```
python dist_binclass.py --test
```

Trainset has 7,000,000 instances (10,938 batches with 32 instances on 2 trainers)
Testset has 1,000,000 instances (100 batches on a single trainer with 1000 instances)

On GCP:
  * With GPU:
    - 1:11 mins per epoch
    - At the end of 2 epochs
      - Train: loss=0.207 acc=0.945
      - Val: loss=0.191 acc=0.953
      - Test accuracy: 0.952
"""

import os
from pathlib import Path

import click
import torch as t
import torch.distributed as dist
import torchmetrics as tm
from bindata import BinDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

DATAROOT = "file://" + str(Path.home() / "mldata" / "binclass" / "1M")
# DATAROOT = "gs://avilabs-mldata/binclass"
N_PARTS = 10
DEVICE = None
# CHECKPOINT = "/home/avilay/mlruns/dist_binclass_net.ckpt"
CHECKPOINT = str(Path.home() / "mlruns" / "dist_binclass_net.ckpt")

# Hyperparams
# BATCH_SIZE = 32
BATCH_SIZE = 16  # Halving the batch size to run on 2 GPUs
LR = 0.003
# N_EPOCHS = 10
N_EPOCHS = 3


def dist_print(text):
    rank = dist.get_rank()
    pid = os.getpid()
    print(f"[{rank}]({pid}): {text}")


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(20, 128)
        self.fc2 = t.nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = t.nn.functional.relu(x)
        x = self.fc2(x)
        y_hat = t.sigmoid(x)
        return y_hat.squeeze(dim=1)


def build_train_datasets(dataroot):
    if dist.get_rank() == 0:
        trainsets, valsets = BinDataset.load_train_val_partitioned(
            DATAROOT, range(N_PARTS // 2)
        )
    else:
        trainsets, valsets = BinDataset.load_train_val_partitioned(
            DATAROOT, range(N_PARTS // 2, N_PARTS)
        )
    trainset = t.utils.data.ConcatDataset(trainsets)
    valset = t.utils.data.ConcatDataset(valsets)
    return trainset, valset


def training_loop(model, optim, loss_fn, traindl, valdl):
    for epoch in range(1, N_EPOCHS + 1):
        train_acc = tm.Accuracy().to(DEVICE)
        train_loss = tm.MeanMetric().to(DEVICE)

        model.train()
        with t.enable_grad():
            for inputs, targets in tqdm(traindl):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                # The standard 5-step training process
                optim.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets.to(t.float32))
                loss.backward()
                optim.step()

                train_acc(outputs, targets)
                train_loss(loss.detach().item())
        train_acc_value = train_acc.compute().item()
        train_loss_value = train_loss.compute().item()

        val_acc = tm.Accuracy(compute_on_step=False).to(DEVICE)
        val_loss = tm.MeanMetric(compute_on_step=False).to(DEVICE)

        model.eval()
        with t.no_grad():
            for inputs, targets in valdl:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets.to(t.float32))

                val_acc(outputs, targets)
                val_loss(loss.detach().item())
        val_acc_value = val_acc.compute().item()
        val_loss_value = val_loss.compute().item()

        print(f"Epoch: {epoch}")
        print(f"Train: loss={train_loss_value:.3f} acc={train_acc_value:.3f}")
        print(f"Val: loss={val_loss_value:.3f} acc={val_acc_value:.3f}")
        print("\n")


def evaluate(model, testdl):
    acc = tm.Accuracy(compute_on_step=False).to(DEVICE)
    model.eval()
    with t.no_grad():
        for inputs, targets in tqdm(testdl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            acc(outputs, targets)
    acc_value = acc.compute().item()
    print(f"Test accuracy: {acc_value:.3f}")


@click.command()
@click.option(
    "--rank", help="Rank of the trainer starting with 0.", type=int, default=0
)
@click.option("--train/--test", help="Whether to train or test.", required=True)
def main(rank, train):
    # Training is done on a 2 GPU cluster. Evaluation is done on a single GPU.
    global DEVICE
    DEVICE = t.device(f"cuda:{rank}")

    if train:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = "2"
        dist.init_process_group("nccl")

        trainset, valset = build_train_datasets(DATAROOT)
        traindl = t.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
        )
        valdl = t.utils.data.DataLoader(
            valset, batch_size=1000, shuffle=False, pin_memory=True
        )

        with t.cuda.device(DEVICE):
            net = Net().to(DEVICE)
            ddp_net = DDP(net, device_ids=[DEVICE], output_device=DEVICE)
            optim = t.optim.Adam(ddp_net.parameters(), lr=LR)
            loss_fn = t.nn.BCELoss()
            training_loop(ddp_net, optim, loss_fn, traindl, valdl)
            dist.barrier()
            if rank == 0:
                # Saving the inner net without DDP wrapper because evaluation will without
                # distributed cluster
                t.save(net, CHECKPOINT)
                print(f"Model saved to {CHECKPOINT}")
        print("Training complete.")
        dist.destroy_process_group()
    else:
        testset = BinDataset.load_test_single(DATAROOT)
        testdl = t.utils.data.DataLoader(
            testset, batch_size=1000, shuffle=False, pin_memory=True
        )
        net = t.load(CHECKPOINT)
        evaluate(net, testdl)


if __name__ == "__main__":
    main()
