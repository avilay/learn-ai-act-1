"""
Run this program with the `-W ignore` to disable all the useless warnings
thrown by Lightning.
```
python -W ignore simple_tuner.py
```
"""
import logging
from pathlib import Path

from data import get_vocab
from models import Simple
from tune import tune
from haikunator import Haikunator
from cprint import info_print

logging.basicConfig(level=logging.ERROR)
logging.getLogger("ax.service.utils.instantiation").setLevel(logging.ERROR)
logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.ERROR)
logging.getLogger("ax.service.managed_loop").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

run_name = Haikunator().haikunate()
dataroot = Path.home() / "mldata"
runroot = Path.home() / "mlruns" / "imdb" / run_name
runroot.mkdir()

vocab, _ = get_vocab(dataroot)

hparams = [
    {"name": "max_epochs", "type": "range", "value_type": "int", "bounds": [3, 10]},
    {
        "name": "max_seq_len",
        "type": "choice",
        "is_ordered": True,
        "value_type": "int",
        "values": [100, 200, 500],
    },
    {
        "name": "batch_size",
        "type": "choice",
        "is_ordered": True,
        "value_type": "int",
        "values": [16, 32, 64],
    },
    {
        "name": "embedding_dim",
        "type": "choice",
        "is_ordered": True,
        "value_type": "int",
        "values": [50, 100, 200, 300],
    },
    {
        "name": "learning_rate",
        "type": "range",
        "value_type": "float",
        "bounds": [1e-6, 1e-2],
        "log_scale": True,
    },
]

info_print(f"\nStarting run {run_name}\n")
tune(
    model_factory=Simple,
    runroot=runroot,
    dataroot=dataroot,
    vocab=vocab,
    total_trials=10,
    hparams=hparams,
)

"""
Best Params: {'max_epochs': 9, 'learning_rate': 0.00158223651599205, 'max_seq_len': 100, 'batch_size': 8, 'embedding_dim': 25}
Values: ({'accuracy': 0.7379166483879089}, {'accuracy': {'accuracy': 0.0}})
"""
