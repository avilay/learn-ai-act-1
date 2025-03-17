import os.path as path
import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
from tqdm import tqdm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.set_printoptions(precision=3)

logformat = "[%(levelname)s %(asctime)s] %(process)s-%(name)s: %(message)s"
logging.basicConfig(format=logformat, level=logging.DEBUG, datefmt="%m-%d %I:%M:%S")


class ImdbReviews(tf.keras.utils.Sequence):
    def __init__(self, dataroot, batch_size=32, split="train", max_seq_len=250, encoder=None):
        if split == "train":
            filelist = path.join(dataroot, "train.csv")
        elif split == "val":
            filelist = path.join(dataroot, "val.csv")
        elif split == "test":
            filelist = path.join(dataroot, "test.csv")
        else:
            raise ValueError(f"Unknown split {split}!")

        if not path.exists(filelist):
            raise RuntimeError("Need to run the dataprep.ipynb before!")

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.encoder = encoder

        target_map = {"pos": 1.0, "neg": 0.0}

        files = []
        with open(filelist, "rt") as f:
            for line in f:
                filepath, label = line.strip().split(",")
                target = target_map[label]
                files.append((filepath, target))
        np.random.shuffle(files)
        self._num_batches = int(np.ceil(len(files) / batch_size))
        logging.info(f"Total of {len(files)} files in {self._num_batches} batches.")

        texts = []
        targets = []
        vocab = set()
        tokenizer = tfds.features.text.Tokenizer()
        for file, target in tqdm(files):
            with open(file, "rt") as f:
                text = f.read()
                text = BeautifulSoup(text, "html.parser").get_text()
                text = text.lower()
                texts.append(text)
                targets.append(target)
                tokens = tokenizer.tokenize(text)
                vocab.update(tokens)

        logging.info("Encoding all the text.")
        encoded_texts = []
        if self.encoder is None:
            self.encoder = tfds.features.text.TokenTextEncoder(vocab, tokenizer=tokenizer)
        vocab_size = len(self.encoder.tokens)
        for text in texts:
            encoded_text = np.array(self.encoder.encode(text))
            encoded_text = encoded_text[encoded_text <= vocab_size]  # Drop OOV words
            encoded_texts.append(encoded_text)

        padded_encoded_texts = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_texts, max_seq_len
        )
        self._encoded_texts = tf.constant(padded_encoded_texts)
        self._targets = tf.reshape(tf.constant(targets), (-1, 1))
        logging.debug(
            f"encoded texts shape={self._encoded_texts.shape}, targets shape={self._targets.shape}"
        )

    def __len__(self):
        return self._num_batches

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_x = self._encoded_texts[start_idx:end_idx]
        batch_y = self._targets[start_idx:end_idx]
        return batch_x, batch_y

    def label(self, target):
        if target == 0:
            return "neg"
        elif target == 1:
            return "pos"
        else:
            raise ValueError(f"Unknown target {target}!")
