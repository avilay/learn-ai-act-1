from pathlib import Path
from enum import Enum, auto
import warnings

import click
import pytorch_lightning as pl
import torch as t
from torchmetrics import Accuracy

from ..data import ImdbDataModule, get_vocab
from .models import create_model

DATAROOT = Path.home() / "mldata"
RUNROOT = Path.home() / "mlruns" / "imdb"
t.set_printoptions(precision=8)


class LoggerType(Enum):
    TB = auto()
    CSV = auto()


class LitImdb(pl.LightningModule):
    def __init__(
        self,
        model_name,
        **kwargs,
    ):
        super().__init__()
        # self.simple = model_factory(vocab_size, max_seq_len, embedding_dim)
        self.model = create_model(model_name, **kwargs)
        self.loss_fn = t.nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy("binary")
        self.val_acc = Accuracy("binary")

        self.val_acc_value = 0.0

        self.save_hyperparameters(ignore=["vocab_size"])

    def configure_optimizers(self):
        return t.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)  # type: ignore

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.train_acc(outputs, targets)

        self.log("train_step_loss", loss)
        self.log("loss", {"train_loss": loss}, on_step=False, on_epoch=True)
        self.log("acc", {"train_acc": self.train_acc}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.val_acc.update(outputs, targets)

        # By default this is accumulated at every step and logged at epoch end
        self.log("loss", {"val_loss": loss})
        self.log("acc", {"val_acc": self.val_acc})

        # logging this separately because it is needed during ax tuning
        self.log("val_acc", self.val_acc)
        return {}


def train(
    logger_type,
    model_name,
    max_epochs,
    max_seq_len,
    batch_size,
    embedding_dim,
    learning_rate,
):
    vocab, _ = get_vocab(DATAROOT)

    print("Instantiating model")
    model = LitImdb(
        model_name=model_name,
        vocab_size=len(vocab),
        max_seq_len=max_seq_len,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
    )

    print("Instantiating data module")
    dm = ImdbDataModule(
        dataroot=DATAROOT, vocab=vocab, max_seq_len=max_seq_len, batch_size=batch_size
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("Instantiating trainer")
        if logger_type == LoggerType.CSV:
            logger = pl.loggers.CSVLogger(save_dir=RUNROOT.parent, name=RUNROOT.name)  # type: ignore
        else:
            logger = True
        trainer = pl.Trainer(
            default_root_dir=RUNROOT,
            max_epochs=max_epochs,
            logger=logger,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
            # MPS does not work because aten::unique op is not implemented
            # accelerator='mps',
            # devices=1
            accelerator="gpu" if t.cuda.is_available() else "cpu",
            devices=1,
        )

        print("Start training")
        trainer.fit(model, dm)
        trainer.validate(datamodule=dm)


@click.command()
@click.option("--logger-type", type=click.Choice(["CSV", "TB"], case_sensitive=False))
@click.option(
    "--model-name", type=click.Choice(["Simple", "Glove"], case_sensitive=False)
)
@click.option("--max-epochs", default=2, help="Number of epochs to train for.")
@click.option(
    "--max_seq_len", default=100, help="Max number of tokens(words) in a review."
)
@click.option("--batch-size", default=32, help="Batch size.")
@click.option(
    "--embedding-dim", default=25, help="Size of vector representing a single token."
)
@click.option(
    "--learning-rate", default=0.0001, help="Learning rate for Adam optimizer."
)
def main(
    logger_type,
    model_name,
    max_epochs,
    max_seq_len,
    batch_size,
    embedding_dim,
    learning_rate,
):
    print(f"Starting training {model_name} model with the following hyperparams:")
    print(f"\tmax_epochs: {max_epochs}")
    print(f"\tmax_seq_len: {max_seq_len}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\tembedding_dim: {embedding_dim}")
    print(f"\tlearning_rate: {learning_rate}")
    if logger_type.upper() == "CSV":
        logger_type = LoggerType.CSV
    else:
        logger_type = LoggerType.TB
    train(
        logger_type,
        model_name,
        max_epochs,
        max_seq_len,
        batch_size,
        embedding_dim,
        learning_rate,
    )


if __name__ == "__main__":
    main()
