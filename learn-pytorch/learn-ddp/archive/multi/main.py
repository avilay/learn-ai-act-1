import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

import hydra
import torch as t
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import multi.mnist as mnist

from .trainer import train

logger = logging.getLogger(__name__)

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


@hydra.main(config_path="./conf", config_name="config")
def main(cfg):
    os.environ["MASTER_ADDR"] = cfg.dist.master.addr
    os.environ["MASTER_PORT"] = str(cfg.dist.master.port)
    os.environ["RANK"] = str(cfg.dist.rank)
    os.environ["WORLD_SIZE"] = str(cfg.dist.world)

    dist.init_process_group("gloo")
    tmpdir = tempfile.gettempdir()
    trainfile = Path(tmpdir) / "trainset.pkl"
    valfile = Path(tmpdir) / "valset.pkl"
    if cfg.dist.rank == 0:
        trainset, valset = mnist.build_train_datasets(cfg.app.datapath)
        with open(trainfile, "wb") as f:
            pickle.dump(trainset, f)
        with open(valfile, "wb") as f:
            pickle.dump(valset, f)
        logger.debug(f"Pickled datasets into {trainfile} and {valfile}.")
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
        batch_size=cfg.app.hparams.batch_size,
        sampler=DistributedSampler(trainset, shuffle=True),
    )
    valdl = DataLoader(
        valset,
        batch_size=min(1000, len(valset)),
        sampler=DistributedSampler(valset, shuffle=False),
    )

    model = DDP(mnist.Net(cfg.app.hparams).to(DEVICE))
    MetricsAccumulator = mnist.BatchMetricsAccumulator
    loss_fn = t.nn.NLLLoss()
    model = train(
        traindl,
        valdl,
        model,
        MetricsAccumulator,
        loss_fn,
        cfg.log_frequency,
        cfg.app.hparams,
    )

    dist.barrier()

    if cfg.dist.rank == 0:
        checkpoint = "model.ckpt"
        logger.info(f"Saving model to {os.getcwd()}/{checkpoint}")
        t.save(model, checkpoint)
        logger.info("Deleting pickled datasets.")
        os.remove(trainfile)
        os.remove(valfile)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
