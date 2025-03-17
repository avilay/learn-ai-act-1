import os.path as path
import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import unicodedata
import re

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.set_printoptions(precision=3)

logformat = "[%(levelname)s %(asctime)s] %(process)s-%(name)s: %(message)s"
logging.basicConfig(format=logformat, level=logging.DEBUG, datefmt="%m-%d %I:%M:%S")


def unicode_to_ascii(string):
    # Normalize accented chars to a uniform representation
    norm_string = unicodedata.normalize("NFD", string)

    # Get rid of accents
    return "".join(char for char in norm_string if unicodedata.category(char) != "Mn")


def prep(sentence):
    sentence = sentence.lower().strip()
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.rstrip().strip()
    sentence = "<start> " + sentence + " <end>"
    return sentence


class EngSpa(tf.keras.utils.Sequence):
    def __init__(self, dataroot, batch_size=32, split="train", eng_encoder=None, spa_encoder=None):
        if split == "train":
            file = path.join(dataroot, "train.txt")
        elif split == "val":
            file = path.join(dataroot, "val.txt")
        else:
            raise ValueError(f"Unknown split {split}!")

        if not path.exists(file):
            raise RuntimeError("Need to run dataprep.ipynb first!")

        self.batch_size = batch_size
        self.eng_encoder = eng_encoder
        self.spa_encoder = spa_encoder

        lines = []
        eng_vocab = set()
        spa_vocab = set()
        tokenizer = tfds.features.text.Tokenizer()
        with open(file, "rt", encoding="utf-8") as f:
            for line in f:
                eng, spa = line.split("\t")
                eng = prep(eng)
                spa = prep(spa)

                lines.append((eng, spa))
        np.random.shuffle(lines)
        self._num_batches = int(np.ceil(len(lines) / batch_size))
        logging.info(f"Total of {len(lines)} lines in {self._num_batches} batches.")

