from pathlib import Path

import pretty_traceback
import pytorch_lightning as pl
import torch as t
import torchvision as tv
from pytorch_lightning.loggers import WandbLogger
import msg
from torchmetrics import Accuracy

pretty_traceback.install()


# Hyperparams
N_EPOCHS = 250
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
Z_DIM = 64
GEN_HIDDEN_DIM = 128
DISC_HIDDEN_DIM = 128

IMG_DIM = 28 * 28
DATAROOT = Path.home() / "mldata" / "mnist"
RUNROOT = Path.home() / "mlruns" / "lit_mnist_gan"


def is_dist():
    return t.cuda.is_available() and t.distributed.is_available()


def gen_block(input_dim, output_dim):
    return t.nn.Sequential(
        t.nn.Linear(input_dim, output_dim),
        t.nn.BatchNorm1d(output_dim),
        t.nn.ReLU(inplace=True),
    )


class Generator(t.nn.Module):
    def __init__(self, z_dim, hidden_dim, im_dim):
        super().__init__()
        self.gen = t.nn.Sequential(
            gen_block(z_dim, hidden_dim),
            gen_block(hidden_dim, hidden_dim * 2),
            gen_block(hidden_dim * 2, hidden_dim * 4),
            gen_block(hidden_dim * 4, hidden_dim * 8),
            t.nn.Linear(hidden_dim * 8, im_dim),
            t.nn.Sigmoid(),
        )

    def forward(self, batch_noise):
        return self.gen(batch_noise)


def disc_block(input_dim, output_dim):
    return t.nn.Sequential(t.nn.Linear(input_dim, output_dim), t.nn.LeakyReLU(0.2))


class Discriminator(t.nn.Module):
    def __init__(self, im_dim, hidden_dim):
        super().__init__()
        self.disc = t.nn.Sequential(
            disc_block(im_dim, hidden_dim * 4),
            disc_block(hidden_dim * 4, hidden_dim * 2),
            disc_block(hidden_dim * 2, hidden_dim),
            t.nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_images):
        return self.disc(batch_images)


class MnistDataModule(pl.LightningDataModule):
    def __init__(self, dataroot, batch_size):
        super().__init__()
        self._trainset = None
        self._dataroot = dataroot
        self.save_hyperparameters(ignore=["dataroot"])

    def prepare_data(self):
        tv.datasets.MNIST(self._dataroot, train=True, download=True)

    def setup(self, stage):
        self._trainset = t.utils.data.DataLoader(
            tv.datasets.MNIST(
                self._dataroot,
                train=True,
                download=False,
                transform=tv.transforms.ToTensor(),
            ),
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def train_dataloader(self):
        return self._trainset


class GAN(pl.LightningModule):
    def __init__(
        self,
        z_dim,
        gen_hidden_dim,
        disc_hidden_dim,
        img_dim,
        gen_lr,
        disc_lr,
        runroot,
        save_freq,
    ):
        super().__init__()
        self.gen = Generator(z_dim, gen_hidden_dim, img_dim)
        self.disc = Discriminator(img_dim, disc_hidden_dim)
        self.save_hyperparameters(ignore=["runroot", "save_freq"])
        self._runroot = runroot
        self._runroot.mkdir(exist_ok=True)
        self._save_freq = save_freq
        self.loss_fn = t.nn.BCEWithLogitsLoss()
        self._val_z = t.randn(25, z_dim)
        self.real_acc = Accuracy()
        self.fake_acc = Accuracy()

    def forward(self, inputs):
        z = t.randn(len(inputs), self.hparams.z_dim)
        return self.gen(z)

    def configure_optimizers(self):
        disc_optim = t.optim.Adam(self.disc.parameters(), lr=self.hparams.disc_lr)
        gen_optim = t.optim.Adam(self.gen.parameters(), lr=self.hparams.gen_lr)
        return [disc_optim, gen_optim]

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            return self._disc_training_step(batch, batch_idx)
        elif optimizer_idx == 1:
            return self._gen_training_step(batch, batch_idx)

    def on_train_epoch_end(self):
        if self.current_epoch % self._save_freq == 0:
            z = self._val_z.to(self.device)
            outputs = self.gen(z)
            images = outputs.detach().cpu().view(-1, 1, 28, 28)
            grid = tv.utils.make_grid(images, nrow=5, padding=3, pad_value=1)
            self.logger.log_image(
                "generated_images",
                images=[grid],
                caption=[f"epoch-{self.current_epoch}"],
            )

    def _disc_training_step(self, batch, batch_idx):
        real_images, _ = batch
        batch_size = len(real_images)
        real_images = real_images.view(batch_size, -1)

        z = t.randn(batch_size, self.hparams.z_dim).type_as(real_images)
        fake_images = self.gen(z)
        fake_outputs = self.disc(fake_images.detach())
        fake_targets = t.zeros_like(fake_outputs)
        fake_loss = self.loss_fn(fake_outputs, fake_targets)
        self.fake_acc(fake_outputs, fake_targets.detach().to(t.int))

        real_outputs = self.disc(real_images)
        real_targets = t.ones_like(real_outputs)
        real_loss = self.loss_fn(real_outputs, real_targets)
        self.real_acc(real_outputs, real_targets.detach().to(t.int))

        loss = (fake_loss + real_loss) / 2
        self.log("disc_loss", loss)
        self.log("fake_acc", self.fake_acc)
        self.log("real_acc", self.real_acc)
        return loss

    def _gen_training_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        z = t.randn(batch_size, self.hparams.z_dim).type_as(batch[0])
        fake_images = self.gen(z)
        fake_outputs = self.disc(fake_images)
        targets = t.ones_like(fake_outputs)
        loss = self.loss_fn(fake_outputs, targets)
        self.log("gen_loss", loss)
        return loss


def main():
    model = GAN(
        Z_DIM,
        GEN_HIDDEN_DIM,
        DISC_HIDDEN_DIM,
        IMG_DIM,
        LEARNING_RATE,
        LEARNING_RATE,
        RUNROOT,
        10,
    )
    dm = MnistDataModule(DATAROOT, BATCH_SIZE)
    logger = WandbLogger(project="lit_mnist_gan", save_dir=RUNROOT, log_model="all")
    logger.watch(model, log="all")
    trainer = pl.Trainer(
        default_root_dir=RUNROOT,
        max_epochs=N_EPOCHS,
        logger=logger,
        gpus=-1 if is_dist() else None,
        strategy="ddp" if is_dist() else None,
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    outputs = model.gen(model._val_z)
    images = outputs.detach().cpu().view(-1, 1, 28, 28)
    grid = tv.utils.make_grid(images, nrow=5, padding=3, pad_value=1)
    logger.log_image("generated_images", images=[grid], caption=["final"])
    msg.sms("+12066173488", "GAN training complete.")


if __name__ == "__main__":
    main()
