import logging
import os
from pathlib import Path

import hydra
import torch as t
from snippets.color_printer import print_error, print_success
from torch.utils.data import DataLoader

from .data import build_dataset
from .model import BinClassifier
from .trainer import test, train

logger = logging.getLogger(__name__)


@hydra.main(config_path="..", config_name="binclass")
def main(cfg):
    logger.info(f"Starting run {cfg.app.name}")

    trainset, valset, testset = build_dataset(cfg.app.dataroot)

    if cfg.app.task == "train":
        traindl = DataLoader(trainset, cfg.train.hparams.batch_size, shuffle=True)
        valdl = DataLoader(valset, len(valset))
        logger.info("Starting training.")
        model = train(cfg.train, traindl, valdl, BinClassifier)
        checkpoint = "model.ckpt"
        logger.info(f"Saving model to {os.getcwd()}/{checkpoint}")
        t.save(model, checkpoint)
    elif cfg.app.task == "test":
        checkpoint = Path(cfg.test.checkpoint)
        if not checkpoint.exists:
            print_error(f"{cfg.test.checkpoint} does not exist!")
        model = t.load(checkpoint)
        testdl = DataLoader(testset, len(testset))
        accuracy = test(testdl, model)
        print_success(f"Test: accuracy = {accuracy:.3f}")


if __name__ == "__main__":
    main()
