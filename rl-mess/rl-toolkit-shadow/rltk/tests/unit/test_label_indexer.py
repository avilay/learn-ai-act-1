import uuid
from rltk.label_indexer import LabelIndexer


def test_len():
    labels = ['up', 'up', 'down', 'strange', 'charmed', 'strange', 'top', 'bottom']
    lblidx = LabelIndexer(labels)
    assert len(set(labels)) == len(lblidx)

    labels = []
    lblidx = LabelIndexer(labels)
    assert 0 == len(lblidx)


def test_get_idx():
    labels = ['up', 'down', 'top']
    lblidx = LabelIndexer(labels)
    for idx in range(len(labels)):
        assert lblidx[idx] in labels


def test_get_lbl():
    labels = ['up', 'down', 'top']
    lblidx = LabelIndexer(labels)
    for label in labels:
        assert 0 <= lblidx[label] < len(labels)


def test_contains():
    labels = ['up', 'down', 'top']
    lblidx = LabelIndexer(labels)
    for label in labels:
        assert label in lblidx
    assert str(uuid.uuid4()) not in lblidx


def test_iter():
    labels = ['up', 'down', 'top']
    lblidx = LabelIndexer(labels)
    for idx, lbl in lblidx:
        assert lbl in labels


def test_correctness():
    labels = ['up', 'down', 'top']
    lblidx = LabelIndexer(labels)
    lbls = []
    idxs = []
    for idx, lbl in lblidx:
        idxs.append(idx)
        lbls.append(lbl)

    for i, idx in enumerate(idxs):
        exp_lbl = lbls[i]
        assert exp_lbl == lblidx[idx]

    for i, lbl in enumerate(lbls):
        exp_idx = idxs[i]
        assert exp_idx == lblidx[lbl]
