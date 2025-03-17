import warnings
from enum import Enum, auto
from pathlib import Path

import click
import pytorch_lightning as pl
import torch as t
import pretty_traceback
from imdb.data import ImdbDataModule, get_vocab
import torchtext as tt

# from ..model import LitImdb
from imdb.model import LitImdb

from .model import SimpleGlove

pretty_traceback.install()

DATAROOT = Path.home() / "mldata"
RUNROOT = Path.home() / "mlruns" / "imdb"
t.set_printoptions(precision=8)


class LoggerType(Enum):
    TB = auto()
    CSV = auto()


def train(
    logger_type,
    max_epochs,
    max_seq_len,
    batch_size,
    embedding_dim,
    learning_rate,
):
    vocab, _ = get_vocab(DATAROOT)

    print("Instantiating model")
    glove_vecs = tt.vocab.GloVe(name="6B", dim=embedding_dim, cache=DATAROOT / "glove")
    imdb_vecs = glove_vecs.get_vecs_by_tokens(vocab.get_itos())
    simple_glove = SimpleGlove(imdb_vecs, max_seq_len)
    model = LitImdb(simple_glove, learning_rate, max_seq_len, embedding_dim)

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
@click.option(
    "--logger-type",
    default="CSV",
    show_default=True,
    type=click.Choice(["CSV", "TB"], case_sensitive=False),
)
@click.option(
    "--max-epochs", default=2, show_default=True, help="Number of epochs to train for."
)
@click.option(
    "--max_seq_len",
    default=100,
    show_default=True,
    help="Max number of tokens(words) in a review.",
)
@click.option("--batch-size", default=32, show_default=True, help="Batch size.")
@click.option(
    "--embedding-dim",
    default=50,
    show_default=True,
    type=click.Choice(["50", "100", "200", "300"]),
    help="Size of vector representing a single token.",
)
@click.option(
    "--learning-rate",
    default=0.0001,
    show_default=True,
    help="Learning rate for Adam optimizer.",
)
def main(
    logger_type,
    max_epochs,
    max_seq_len,
    batch_size,
    embedding_dim,
    learning_rate,
):
    print("Starting training Simple model with the following hyperparams:")
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
        max_epochs,
        max_seq_len,
        batch_size,
        embedding_dim,
        learning_rate,
    )


if __name__ == "__main__":
    main()
