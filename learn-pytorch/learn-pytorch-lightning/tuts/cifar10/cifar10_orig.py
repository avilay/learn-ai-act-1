"""
https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/cifar10-baseline.html

1. Why does the functional metric work with distirbuted? Is there some more magic inside the self.log for this?
2. Why won't the learning rate monitor log only from step 49 to step 199 instead of from step 0 to step 236?

It is ok to overestimate the number of steps for the learning rate scheduler. In reality there are only 40,000
training data points, but I am still using 45,000 when calculating the number of steps so that the scheduler won't
"run out" of steps.

"""
import os
from pathlib import Path

# import msg
# from distml import dprint
import pytorch_lightning as pl
import torch as t
import torch.nn.functional as F
import torchvision as tv
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from torchmetrics import Accuracy
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import accuracy
import pretty_traceback

pretty_traceback.install()

# Hyperparams
# BATCH_SIZE = 64
BATCH_SIZE = 256  # if GPUs
LEARNING_RATE = 0.05
N_EPOCHS = 20
RUNROOT = Path.home() / "mlruns" / "cifar10_orig"
DATAROOT = Path.home() / "mldata" / "cifar10"
NUM_WORKERS = int(os.cpu_count() / 2)

# pl.seed_everything(7)


def build_data():
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
        data_dir=DATAROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
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
    def __init__(self, learning_rate, model=None):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model if model else build_model()
        self.val_acc = Accuracy(num_classes=10, compute_on_step=False)
        self.test_acc = Accuracy(num_classes=10, compute_on_step=False)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        # dprint(f"Epoch {self.current_epoch}: batch {batch_idx} {y.shape}")
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = t.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            if stage == "val":
                self.val_acc(preds, y)
                self.log("val_acc2", self.val_acc)
            elif stage == "test":
                self.test_acc(preds, y)
                self.log("test_acc2", self.test_acc)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optim = t.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=5e-4,
        )
        # steps_per_epoch = 45_000 // BATCH_SIZE
        if t.cuda.is_available() and t.distributed.is_available():
            steps_per_epoch = 45_000 // (t.cuda.device_count() * BATCH_SIZE)
        else:
            steps_per_epoch = 45_000 // BATCH_SIZE
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


def main():
    dist = t.cuda.is_available() and t.distributed.is_available()
    logger = WandbLogger(project="cifar10_orig", save_dir=RUNROOT, log_model="all",)
    dm = build_data()
    model = LitResnet(LEARNING_RATE)
    trainer = pl.Trainer(
        default_root_dir=RUNROOT,
        max_epochs=N_EPOCHS,
        logger=logger,
        gpus=-1 if dist else None,
        strategy="ddp_find_unused_parameters_false" if dist else None,
        callbacks=[
            TQDMProgressBar(refresh_rate=5),
            LearningRateMonitor(logging_interval="step"),
        ],
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    # msg.sms(tonum="+12066173488", content="CIFAR10 training complete.")


if __name__ == "__main__":
    main()
