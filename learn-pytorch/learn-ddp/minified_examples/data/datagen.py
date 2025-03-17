import sys
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import hydra
from cprint import danger_print

from bindata import BinDataset, build_url


@hydra.main(config_path=".", config_name="datagen")
def main(cfg):
    flds = urlsplit(cfg.outurl)
    if flds.scheme not in ["file", "gs"]:
        danger_print("Only file:// and gs:// type urls are supported.")
        sys.exit(1)

    if cfg.n_parts <= 0:
        danger_print("Can only specify positive number of parts.")

    url = partial(build_url, flds.scheme, flds.netloc)
    path = Path(flds.path)

    names = ["train", "val", "test"]
    filenames = [url(path / f"{name}.csv") for name in names]
    datasets = BinDataset.generate(
        n_samples=cfg.n_samples,
        train_split=cfg.train_split,
        test_split=cfg.test_split,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=cfg.n_repeated,
        flip_y=cfg.flip_y,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
    )
    [ds.save(fn) for ds, fn in zip(datasets, filenames)]

    if cfg.n_parts > 1:
        for dataset, name in zip(datasets, names):
            parts = dataset.partition(cfg.n_parts)
            for i, part in enumerate(parts):
                fn = url(path / f"{name}_part_{i:02d}.csv")
                part.save(fn)


if __name__ == "__main__":
    main()
