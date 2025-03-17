import logging
from pathlib import Path

import click
import pretty_traceback
import pytorch_lightning as pl
import torch as t
from ax import optimize
from cprint import cprint, success_print
from haikunator import Haikunator
from pytorch_lightning.callbacks import ModelCheckpoint

from ..data import ImdbDataModule, get_vocab
from .train import LitImdb

pretty_traceback.install()

DATAROOT = Path.home() / "mldata"
RUNROOT = Path.home() / "mlruns" / "imdb"
t.set_printoptions(precision=8)

logging.basicConfig(level=logging.ERROR)
logging.getLogger("ax.service.utils.instantiation").setLevel(logging.ERROR)
logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.ERROR)
logging.getLogger("ax.service.managed_loop").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# To be filled in by caller or main
hparams_spec = [
    {"name": "max_epochs", "type": "range", "value_type": "int", "bounds": []},
    {
        "name": "max_seq_len",
        "type": "choice",
        "is_ordered": True,
        "value_type": "int",
        "values": [],
    },
    {
        "name": "batch_size",
        "type": "choice",
        "is_ordered": True,
        "value_type": "int",
        "values": [],
    },
    {
        "name": "embedding_dim",
        "type": "choice",
        "is_ordered": True,
        "value_type": "int",
        "values": [],
    },
    {
        "name": "learning_rate",
        "type": "range",
        "value_type": "float",
        "bounds": [],
        "log_scale": True,
    },
]

# defining these so it is easy to set values in the hparams spec dict
# it can be done with hparams[MAX_EPOCHS]["values"] = [1, 2, 3]
# instead of hparams[0]["values"] = [1, 2, 3] which can be error prone.
MAX_EPOCHS = 0
MAX_SEQ_LEN = 1
BATCH_SIZE = 2
EMBEDDING_DIM = 3
LEARNING_RATE = 4


def trial(hparams):
    cprint(1, "\nStarting new trial with the following hyper parameters:")
    for k, v in hparams.items():
        cprint(1, f"{k}: {v}")

    checkpoint_filename = Haikunator().haikunate() + "{epoch}-{iter}.ckpt"
    logger = pl.loggers.CSVLogger(save_dir=RUNROOT.parent, name=RUNROOT.name)  # type: ignore
    trainer = pl.Trainer(
        default_root_dir=RUNROOT,
        max_epochs=hparams["max_epochs"],
        callbacks=[ModelCheckpoint(filename=checkpoint_filename, monitor="val_acc")],
        logger=logger,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1000,
        gpus=1 if t.cuda.is_available() else None,
    )

    vocab, _ = get_vocab(DATAROOT)
    dm = ImdbDataModule(
        dataroot=DATAROOT,
        vocab=vocab,
        max_seq_len=hparams["max_seq_len"],
        batch_size=hparams["batch_size"],
    )
    model = LitImdb(
        model_name="Simple",
        vocab_size=len(vocab),
        max_seq_len=hparams["max_seq_len"],
        embedding_dim=hparams["embedding_dim"],
        learning_rate=hparams["learning_rate"],
    )
    trainer.fit(model, dm)

    val_metrics = trainer.validate(datamodule=dm, verbose=False)
    val_acc = val_metrics[0]["val_acc"]
    return {"accuracy": (val_acc, 0.0)}


def tune(total_trials, hparams):
    vocab, _ = get_vocab(DATAROOT)
    best_params, values, _, _ = optimize(
        hparams,
        evaluation_function=trial,  # type: ignore
        objective_name="accuracy",
        minimize=False,
        total_trials=total_trials,
    )
    success_print(f"Best Params: {best_params}")
    success_print(f"Values: {values}")


@click.command()
@click.option("--total-trials", default=3, help="Number of trials run by Ax.")
@click.option("--max-epochs", default="3,10", help="Number of epochs to train for.")
@click.option(
    "--max-seq-len",
    default="100,200,500",
    help="Max number of tokens(words) in a review.",
)
@click.option("--batch-size", default="16,32,64", help="Batch size.")
@click.option(
    "--embedding-dim",
    default="50,100,200,300",
    help="Size of vector representing a single token.",
)
@click.option(
    "--learning-rate", default="1e-6,1e-2", help="Learning rate for Adam optimizer."
)
def main(
    total_trials, max_epochs, max_seq_len, batch_size, embedding_dim, learning_rate
):
    max_epochs = [int(elem) for elem in max_epochs.split(",")]
    hparams_spec[MAX_EPOCHS]["bounds"] = max_epochs

    max_seq_len = [int(elem) for elem in max_seq_len.split(",")]
    hparams_spec[MAX_SEQ_LEN]["values"] = max_seq_len

    batch_size = [int(elem) for elem in batch_size.split(",")]
    hparams_spec[BATCH_SIZE]["values"] = batch_size

    embedding_dim = [int(elem) for elem in embedding_dim.split(",")]
    hparams_spec[EMBEDDING_DIM]["values"] = embedding_dim

    learning_rate = [float(elem) for elem in learning_rate.split(",")]
    hparams_spec[LEARNING_RATE]["bounds"] = learning_rate

    tune(total_trials, hparams_spec)


if __name__ == "__main__":
    main()
