import os
from pathlib import Path

import click
import pretty_traceback
import pytorch_lightning as pl
import torch as t
import torchvision as tv
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

pretty_traceback.install()

# Hyperparams
LEARNING_RATE = 0.05
N_EPOCHS = 10
BATCH_SIZE_GPU = 256
BATCH_SIZE_CPU = 64

PROJECT = "cifar10"
RUNROOT = Path.home() / "mlruns" / PROJECT
DATAROOT = Path.home() / "mldata" / PROJECT
N_WORKERS = int(os.cpu_count() / 2)


def is_dist():
    return t.cuda.is_available() and t.distributed.is_available()


def build_data(*, dataroot, n_workers, batch_size):
    """
    cifar10_normalization returns a Normalize object
    <class 'torchvision.transforms.transforms.Normalize'> Normalize(mean=[0.4913725490196078, 0.4823529411764706, 0.4466666666666667], std=[0.24705882352941178, 0.24352941176470588, 0.2615686274509804])
    Each batch in the train dataloader will yield a 2 element tuple that looks like -
    torch.Size([64, 3, 32, 32]) torch.Size([64])
    """
    train_xforms = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )

    test_xforms = tv.transforms.Compose(
        [tv.transforms.ToTensor(), cifar10_normalization()]
    )

    dm = CIFAR10DataModule(
        data_dir=dataroot,
        batch_size=batch_size,
        num_workers=n_workers,
        train_transforms=train_xforms,
        test_transforms=test_xforms,
        val_transforms=test_xforms,
    )
    return dm


def build_model():
    """
    The built-in resnet model was trained on ImageNet so its last layer has 1000 output units. This needs to be replaced
    with a different FC layer with 10 output units for 10 CIFAR10 classes. The convenience parameter `num_classes`
    does this for us. Also, I am just interested in the architecture, I don't want the weights, hence setting the
    `pretrained=False`.

    Another change is in the first layer which is a Conv2D that takes in 3 channels and outputs 64 channels. That part
    does not need to change. But the kernel size, padding, and stride length all need to change to account for the
    change in the image dims from 224x224 ImageNet images to 32x32 CIFAR10 images.

    And for some reason the tutorial gets rid of the first max pool layer that comes after
    Conv2D -> BatchNorm -> ReLU -> MaxPool
    This is done by replacing it with an Identity layer, which is a no-op layer.
    """
    model = tv.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = t.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = t.nn.Identity()
    return model


class LitResnet(pl.LightningModule):
    def __init__(self, learning_rate, batch_size, model=None):
        super().__init__()
        self.model = model if model else build_model()

        self.val_acc = Accuracy(num_classes=10, compute_on_step=False)
        self.train_acc = Accuracy(num_classes=10, compute_on_step=False)
        self.test_acc = Accuracy(num_classes=10)

        self.loss_fn = t.nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=["model"])

    def configure_optimizers(self):
        optim = t.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45_000 // self.hparams.batch_size
        scheduler_dict = {
            "scheduler": t.optim.lr_scheduler.OneCycleLR(
                optim,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optim, "lr_scheduler": scheduler_dict}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.train_acc(outputs, targets)

        self.log("train_step_loss", loss)
        self.log("loss", {"train_loss": loss}, on_step=False, on_epoch=True)
        self.log("acc", {"train_acc": self.train_acc}, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        self.val_acc(outputs, targets)
        self.log("loss", {"val_loss": loss})
        self.log("acc", {"val_acc": self.val_acc})

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        self.test_acc(outputs, targets)
        self.log("test_acc", self.test_acc)


@click.command()
@click.option("--train/--test", help="Whether to test or train.")
@click.option("--ckpt", help="Checkpoint path", default="")
def main(train, ckpt):
    batch_size = BATCH_SIZE_GPU if is_dist() else BATCH_SIZE_CPU

    logger = WandbLogger(project=PROJECT, save_dir=RUNROOT, log_model="all",)
    dm = build_data(dataroot=DATAROOT, n_workers=N_WORKERS, batch_size=batch_size,)

    if train:
        model = LitResnet(LEARNING_RATE, batch_size * t.cuda.device_count())
        logger.watch(model, log="all")
        trainer = pl.Trainer(
            default_root_dir=RUNROOT,
            max_epochs=N_EPOCHS,
            callbacks=[
                TQDMProgressBar(refresh_rate=10),
                LearningRateMonitor(logging_interval="step"),
            ],
            logger=logger,
            gpus=-1 if is_dist() else None,
            strategy="ddp_find_unused_parameters_false" if is_dist() else None,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, dm)
    else:
        model = LitResnet.load_from_checkpoint(ckpt)
        logger.watch(model, log="all")
        trainer = pl.Trainer(
            default_root_dir=RUNROOT,
            callbacks=[TQDMProgressBar(refresh_rate=10)],
            logger=logger,
            gpus=[0] if is_dist() else None,
            replace_sampler_ddp=False,
        )
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
