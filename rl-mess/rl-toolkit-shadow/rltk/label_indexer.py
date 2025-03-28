from typing import Union, List, Dict, Sequence


class LabelIndexer:
    def __init__(self, labels: Sequence[str]):
        self._idx2lbl: List[str] = list(set(labels))
        self._lbl2idx: Dict[str, int] = {
            lbl: idx for idx, lbl in enumerate(self._idx2lbl)}

    def __getitem__(self, key: Union[int, str]) -> Union[int, str]:
        if isinstance(key, int):
            return self._idx2lbl[key]
        elif isinstance(key, str):
            return self._lbl2idx[key]
        else:
            raise ValueError()

    def __contains__(self, key: Union[int, str]) -> bool:
        if isinstance(key, int):
            return key < len(self._idx2lbl)
        elif isinstance(key, str):
            return key in self._lbl2idx
        else:
            raise ValueError()

    def __len__(self):
        return len(self._idx2lbl)

    def __iter__(self):
        for idx, label in enumerate(self._idx2lbl):
            yield idx, label
