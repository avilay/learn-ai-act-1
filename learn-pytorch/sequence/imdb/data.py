import os
from collections import OrderedDict, defaultdict
from enum import Enum, auto
from pathlib import Path
import pickle
from typing import Callable
import urllib.request
import subprocess
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.vocab import vocab as vocab_factory
from tqdm import tqdm

DATAURL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def get_vocab(dataroot: Path) -> tuple[Vocab, list[int]]:
    pickled_vocab = Path(".") / "vocab.pkl"

    if not pickled_vocab.exists():
        dirpaths = [
            dataroot / "aclImdb" / "train" / "pos",
            dataroot / "aclImdb" / "train" / "neg",
            dataroot / "aclImdb" / "test" / "pos",
            dataroot / "aclImdb" / "test" / "neg",
        ]
        tokenize = get_tokenizer("basic_english")
        tok_counts: dict[str, int] = defaultdict(int)
        lens = []
        for dirpath in dirpaths:
            print(f"\nReading files from {dirpath}")
            files = os.listdir(dirpath)
            for file in tqdm(files):
                filepath = dirpath / file
                with open(filepath) as f:
                    review = f.read()
                    toks = tokenize(review)
                    lens.append(len(toks))
                    for tok in toks:
                        tok_counts[tok] += 1
        ordered_tok_counts = OrderedDict(
            sorted(tok_counts.items(), key=lambda kv: kv[1], reverse=True)
        )
        vocab = vocab_factory(ordered_dict=ordered_tok_counts)
        with open(pickled_vocab, "wb") as f:  # type: ignore
            pickle.dump((vocab, lens), f)  # type: ignore

    with open(pickled_vocab, "rb") as f:  # type: ignore
        vocab, lens = pickle.load(f)  # type: ignore
    return vocab, lens


class Split(Enum):
    train = auto()
    val = auto()
    test = auto()


class IMDB(Dataset):
    def __init__(
        self,
        root: Path,
        split: Split,
        tokenize: Callable[[str], list[str]],
        vocab: Vocab,
        max_seq_len=10,
    ):
        self._tokenize = tokenize
        self._vocab_stoi = vocab.get_stoi()
        self._max_seq_len = max_seq_len

        imdbroot = root / "aclImdb"
        if not imdbroot.exists():
            raise RuntimeError(
                "Download the data by calling IMDB.download(dataroot) first!"
            )

        if split == Split.train:
            dirpath = imdbroot / "train" / "pos"
            pos_filepaths = [
                dirpath / filename for filename in sorted(os.listdir(dirpath))
            ]
            dirpath = imdbroot / "train" / "neg"
            neg_filepaths = [
                dirpath / filename for filename in sorted(os.listdir(dirpath))
            ]
        elif split == Split.val:
            dirpath = imdbroot / "test" / "pos"
            pos_filepaths = [
                dirpath / filename for filename in sorted(os.listdir(dirpath))[:1200]
            ]
            dirpath = imdbroot / "test" / "neg"
            neg_filepaths = [
                dirpath / filename for filename in sorted(os.listdir(dirpath))[:1200]
            ]
        else:
            dirpath = imdbroot / "test" / "pos"
            pos_filepaths = [
                dirpath / filename for filename in sorted(os.listdir(dirpath))[1200:]
            ]
            dirpath = imdbroot / "test" / "neg"
            neg_filepaths = [
                dirpath / filename for filename in sorted(os.listdir(dirpath))[1200:]
            ]
        self._filepaths = [(filepath, 1) for filepath in pos_filepaths] + [
            (filepath, 0) for filepath in neg_filepaths
        ]

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, index) -> tuple[np.ndarray, int]:
        filepath, label = self._filepaths[index]
        review = ""
        with open(filepath) as f:
            review = f.read()
        toks = self._tokenize(review)
        tokidxs = np.array([self._vocab_stoi[tok] for tok in toks], dtype=np.int32)
        if len(tokidxs) >= self._max_seq_len:
            tokidxs = tokidxs[: self._max_seq_len]
        else:
            tokidxs = np.pad(tokidxs, (0, self._max_seq_len - len(tokidxs)))
        return tokidxs, label

    @staticmethod
    def download(dataroot: Path):
        with urllib.request.urlopen(DATAURL) as response:
            gzcontents = response.read()

        filepath = dataroot / "aclImdb.gz"
        with open(filepath, "wb") as f:
            f.write(gzcontents)

        subprocess.run(["tar", "xf", filepath, "--directory", dataroot])


class ImdbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataroot: Path,
        vocab: Vocab,
        max_seq_len: int,
        batch_size: int,
    ):
        super().__init__()
        self._trainset: None | Dataset = None
        self._valset: None | Dataset = None
        self._testset: None | Dataset = None
        self._dataroot: Path = dataroot
        self._vocab: Vocab = vocab
        self._tokenize = get_tokenizer("basic_english")

        self.save_hyperparameters(ignore=["dataroot", "vocab"])

    def setup(self, stage):
        if stage == "fit" or "validate":
            self._trainset = IMDB(
                self._dataroot,
                Split.train,
                self._tokenize,
                self._vocab,
                self.hparams.max_seq_len,  # type: ignore
            )
            self._valset = IMDB(
                self._dataroot,
                Split.val,
                self._tokenize,
                self._vocab,
                self.hparams.max_seq_len,  # type: ignore
            )
        else:
            self._testset = IMDB(
                self._dataroot,
                Split.test,
                self._tokenize,
                self._vocab,
                self.hparams.max_seq_len,  # type: ignore
            )

    def train_dataloader(self):
        if self._trainset is not None:
            return DataLoader(
                self._trainset, batch_size=self.hparams.batch_size, shuffle=True  # type: ignore
            )
        else:
            raise RuntimeError("setup() was not called!")

    def val_dataloader(self):
        if self._valset is not None:
            return DataLoader(self._valset, batch_size=100, shuffle=False)
        else:
            raise RuntimeError("setup() was not called!")

    def test_dataloader(self):
        if self._testset is not None:
            return DataLoader(self._testset, batch_size=100, shuffle=False)
        else:
            raise RuntimeError("setup() was not called!")
