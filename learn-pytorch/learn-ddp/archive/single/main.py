import logging
import os
import sys
from pathlib import Path

import hydra
import torch as t

import single.binclass as binclass
import single.mnist as mnist

from .trainer import evaluate, train

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="./conf", config_name="config",
)
def main(cfg):
    logger.info(f"Starting {cfg.app.name}.{cfg.cmd}")

    if cfg.app.name == "binclass":
        build_train_datasets = binclass.build_train_datasets
        build_test_dataset = binclass.build_test_dataset
        Net = binclass.Net
        MetricsAccumulator = binclass.BatchMetricsAccumulator
        loss_fn = binclass.MyBCELoss()
    elif cfg.app.name == "mnist":
        build_train_datasets = mnist.build_train_datasets
        build_test_dataset = mnist.build_test_dataset
        Net = mnist.Net
        MetricsAccumulator = mnist.BatchMetricsAccumulator
        loss_fn = t.nn.NLLLoss()
    else:
        print(f"Unknown app name {cfg.app.name}! Valid values are binclass and mnist")
        sys.exit(1)

    datapath = Path.expanduser(Path(cfg.dataroot)) / cfg.app.datadir
    if cfg.cmd == "train":
        trainset, valset = build_train_datasets(datapath)
        model = train(
            trainset,
            valset,
            Net,
            MetricsAccumulator,
            loss_fn,
            cfg.log_frequency,
            cfg.app.hparams,
        )
        checkpoint = "model.ckpt"
        logger.info(f"Saving model to {os.getcwd()}/{checkpoint}")
        t.save(model, checkpoint)
    elif cfg.cmd == "evaluate":
        testset = build_test_dataset(datapath)
        checkpoint = Path(cfg.eval_checkpoint)
        if not checkpoint.exists():
            print(f"{cfg.test.checkpoint} does not exist!")
            sys.exit(1)
        model = t.load(checkpoint)
        accuracy = evaluate(testset, MetricsAccumulator, model)
        print(f"Test: Accuracy = {accuracy:.3f}")
    else:
        print(f"Unknown cmd {cfg.cmd}! Valid values are train and evaluate")
        sys.exit(1)


if __name__ == "__main__":
    main()
