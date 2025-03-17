import importlib
import logging
import os
import pickle
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import torch as t
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as td
import torchmetrics as tm
import torchvision as tv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEVICE = "empty"


# region Model details
class BatchMetricsAccumulator:
    def __init__(self):
        self._losses = []
        self._acc_fn = tm.Accuracy().to(DEVICE)

    def __call__(self, outputs, targets, loss):
        self._losses.append(loss)
        preds = t.argmax(outputs, dim=1)
        self._acc_fn.update(preds, targets)

    def compute(self):
        return sum(self._losses) / len(self._losses), self._acc_fn.compute()


class Net(t.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout2d(hparams.dropouts[0])
        self.dropout2 = t.nn.Dropout2d(hparams.dropouts[1])
        self.fc1 = t.nn.Linear(9216, 128)
        self.fc2 = t.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# endregion Model details

# region Training loop
def _validate(model, loss_fn, valdl, acc_metrics=lambda x, y, z: 0):
    model.eval()
    with t.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valdl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            acc_metrics(outputs, targets, loss.detach().item())


def _train(model, loss_fn, optim, traindl, acc_metrics=lambda x, y, z: 0):
    model.train()
    with t.enable_grad():
        for batch_idx, (inputs, targets) in enumerate(traindl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # The standard 5-step training process
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()

            acc_metrics(outputs, targets, loss.detach().item())


def train(traindl, valdl, model, MetricsAccumulator, loss_fn, log_frequency, hparams):
    logger.info(f"Starting training on {DEVICE}")
    module = importlib.import_module("torch.optim")
    Optim = getattr(module, hparams.optim.class_name)
    optim = Optim(model.parameters(), **hparams.optim.args)

    try:
        for epoch in range(1, hparams.epochs + 1):
            if epoch % log_frequency == 0:
                acc_train_batch_metrics = MetricsAccumulator()
                _train(model, loss_fn, optim, traindl, acc_train_batch_metrics)
                train_loss, train_acc = acc_train_batch_metrics.compute()

                logger.info(
                    f"Epoch {epoch}: Train [loss={train_loss:.3f} acc={train_acc:.3f}]"
                )

                acc_val_batch_metrics = MetricsAccumulator()
                _validate(model, loss_fn, valdl, acc_val_batch_metrics)
                val_loss, val_acc = acc_val_batch_metrics.compute()

                logger.info(
                    f"Epoch {epoch}: Val [loss={val_loss:.3f} acc={val_acc:.3f}]"
                )
                print("\n")
            else:
                _train(model, loss_fn, optim, traindl)
    except KeyboardInterrupt:
        logger.error("Training interrupted!")

    return model


def evaluate(testdl, MetricsAccumulator, model):
    model.eval()

    metrics = MetricsAccumulator()

    logger.info(f"Starting eval on {DEVICE}")
    with t.no_grad():
        for inputs, targets in tqdm(testdl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            metrics(outputs, targets, 0.0)
    _, acc = metrics.compute()
    return acc


# endregion Training loop

# region Data details
def _build_xform():
    means = [0.5]
    stds = [0.5]
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize(means, stds)]
    )
    return xform


def build_train_datasets(datapath):
    xform = _build_xform()
    traindata = tv.datasets.MNIST(datapath, download=True, train=True, transform=xform)
    trainsize = int(0.9 * len(traindata))
    valsize = len(traindata) - trainsize
    trainset, valset = td.random_split(traindata, (trainsize, valsize))
    logger.info(
        f"Prepared trainset with {len(trainset)} examples and valset with {len(valset)} examples."
    )
    return trainset, valset


def build_test_dataset(datapath):
    xform = _build_xform()
    testset = tv.datasets.MNIST(datapath, download=False, train=False, transform=xform)
    logger.info(f"Prepared testset with {len(testset)} examples.")
    return testset


# endregion Data details


@hydra.main(config_path="./conf", config_name="dist-mnist")
def main(cfg):
    os.environ["MASTER_ADDR"] = cfg.dist.master.addr
    os.environ["MASTER_PORT"] = str(cfg.dist.master.port)
    os.environ["RANK"] = str(cfg.dist.rank)
    os.environ["WORLD_SIZE"] = str(cfg.dist.world)
    datapath = Path.home() / "mldata" / "mnist"
    dist.init_process_group("nccl")

    global DEVICE
    DEVICE = t.device(cfg.dist.devices[dist.get_rank()])
    # if DEVICE.type.startswith("cuda"):
    #     print("Setting CUDA_VISIBLE_DEVICES")
    #     os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE.type

    if cfg.cmd == "train":
        tmpdir = tempfile.gettempdir()
        trainfile = Path(tmpdir) / "trainset.pkl"
        valfile = Path(tmpdir) / "valset.pkl"
        if cfg.dist.rank == 0:
            trainset, valset = build_train_datasets(datapath)
            with open(trainfile, "wb") as f:
                pickle.dump(trainset, f)
            with open(valfile, "wb") as f:
                pickle.dump(valset, f)
        dist.barrier()
        if cfg.dist.rank > 0:
            if not trainfile.exists() or not valfile.exists():
                logger.error(f"Could not find dataset {trainfile} or {valfile}!")
                sys.exit(1)
            with open(trainfile, "rb") as f:
                trainset = pickle.load(f)
            with open(valfile, "rb") as f:
                valset = pickle.load(f)

        logger.info(
            f"Loaded trainset with {len(trainset)} examples and valset with {len(valset)} examples."
        )

        traindl = DataLoader(
            trainset,
            batch_size=cfg.hparams.batch_size,
            sampler=DistributedSampler(trainset, shuffle=True),
        )
        valdl = DataLoader(
            valset,
            batch_size=min(1000, len(valset)),
            sampler=DistributedSampler(valset, shuffle=False),
        )

        # t.cuda.set_device(cfg.dist.rank)
        with t.cuda.device(DEVICE):
            model = DDP(
                Net(cfg.hparams).to(DEVICE), device_ids=[DEVICE], output_device=DEVICE
            )

            MetricsAccumulator = BatchMetricsAccumulator
            loss_fn = t.nn.NLLLoss()
            start = datetime.now()
            model = train(
                traindl,
                valdl,
                model,
                MetricsAccumulator,
                loss_fn,
                cfg.log_frequency,
                cfg.hparams,
            )

            dist.barrier()
            end = datetime.now()
            logger.info(f"Training took {end - start}")
            if cfg.dist.rank == 0:
                checkpoint = "model.ckpt"
                logger.info(f"Saving model to {os.getcwd()}/{checkpoint}")
                t.save(model, checkpoint)
                logger.info("Deleting pickled datasets.")
                os.remove(trainfile)
                os.remove(valfile)

            dist.destroy_process_group()
    elif cfg.cmd == "test":
        testset = build_test_dataset(datapath)
        testdl = DataLoader(
            testset,
            batch_size=100,
            sampler=DistributedSampler(testset, shuffle=False),
        )
        checkpoint = Path(cfg.checkpoint)
        map_location = {"cuda:0": f"cuda:{cfg.dist.rank}"}
        # t.cuda.set_device(cfg.dist.rank)
        with t.cuda.device(DEVICE):
            model = t.load(checkpoint, map_location=map_location)
            model.device_ids = [cfg.dist.rank]
            model.output_device = cfg.dist.rank
            accuracy = evaluate(testdl, BatchMetricsAccumulator, model)
            print(f"Test accuracy: {accuracy:.3f}")
    else:
        print("Unknown command.")
        sys.exit(1)


if __name__ == "__main__":
    main()
