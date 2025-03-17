import functools
from typing import List, Optional, Union

import numpy as np
from bokeh.io import show
from bokeh.layouts import row, grid
from bokeh.plotting import Figure, figure
from PIL import Image
from tensorflow import Tensor

ImageType = Union[np.ndarray, Tensor, Image.Image]
TensorType = Union[np.ndarray, Tensor]
LabelIdxType = Union[np.ndarray, Tensor, int]


def _build_image_figure(image: Image.Image, label: str, height: int, width: int) -> Figure:
    p = figure(title=label, plot_height=height, plot_width=width, toolbar_location=None, tools="")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False
    aspect = image.width / image.height
    dw = 10
    dh = dw / aspect
    if image.mode == "L":
        image_func = functools.partial(p.image, x=0, y=0, dh=dh, dw=dw, palette="Greys256")
    else:
        if image.mode == "RGB":
            image = image.convert("RGBA")
        image_func = functools.partial(p.image_rgba, x=0, y=0, dh=dh, dw=dw)
    image = np.array(image)
    image = np.flipud(image)
    image_func([image])
    return p


def _build_prob_figure(
    prob_dist: np.ndarray, target: int, classes: List[str], height: int, width: int
) -> Figure:
    p = figure(
        plot_height=height, plot_width=width, toolbar_location=None, tools="", x_range=classes
    )
    p.axis.major_label_orientation = 1
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    data = dict(classes=classes, prob_dist=prob_dist)
    p.vbar(source=data, x="classes", top="prob_dist", width=0.75, color="brown")

    probs = np.zeros(len(classes))
    probs[target] = 1.0
    data = dict(classes=classes, probs=probs)
    p.vbar(source=data, x="classes", top="probs", width=0.25)

    return p


def _get_image(image: ImageType) -> Image.Image:
    if isinstance(image, Tensor):
        image = image.numpy()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode not in ["L", "RGB", "RGBA"]:
        raise ValueError(f"Unsupported image mode: {image.mode}!")
    return image


def _get_label_idx(label_idx: LabelIdxType) -> int:
    if label_idx is not None:
        if isinstance(label_idx, Tensor):
            label_idx = label_idx.numpy()
        if isinstance(label_idx, np.ndarray):
            if label_idx.shape == ():
                label_idx = int(label_idx)
            else:
                raise ValueError(f"Label index can only be a single number!")
        return label_idx
    else:
        return -1


def _get_label(label_idx: int, classes: Optional[List[str]]) -> str:
    if label_idx == -1:
        return ""
    if classes is None:
        return str(label_idx)
    return classes[label_idx]


def show_image(
    image: ImageType,
    *,
    label_idx: Optional[LabelIdxType] = None,
    classes: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> None:

    image = _get_image(image)
    label_idx = _get_label_idx(label_idx)
    label = _get_label(label_idx, classes)
    height = height if height is not None else image.size[1]
    width = width if width is not None else image.size[0]
    fig = _build_image_figure(image, label, height, width)
    show(fig)


def show_images(
    images: List[ImageType],
    *,
    label_idxs: Optional[List[LabelIdxType]] = None,
    classes: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> None:
    if label_idxs and len(images) != len(label_idxs):
        raise ValueError(f"Cannot match {len(images)} images with {len(label_idxs)} labels!")
    label_idxs = label_idxs if label_idxs else [None] * len(images)
    figs = []
    for image, label_idx in zip(images, label_idxs):
        image = _get_image(image)
        label_idx = _get_label_idx(label_idx)
        label = _get_label(label_idx, classes)
        plot_height = height if height is not None else image.size[1]
        plot_width = width if width is not None else image.size[0]
        fig = _build_image_figure(image, label, plot_height, plot_width)
        figs.append(fig)
    show(row(figs))


def show_image_probs(
    *,
    images: List[ImageType],
    probs: TensorType,
    label_idxs: List[LabelIdxType],
    classes: List[str],
    height=150,
    width=150,
) -> None:
    if len(images) != len(label_idxs):
        raise ValueError(f"Cannot match {len(images)} images with {len(label_idxs)} labels!")
    if len(images) != len(probs):
        raise ValueError(f"Cannot match {len(images)} images with {len(probs)} probs!")

    imgfigs = []
    probfigs = []
    if isinstance(probs, Tensor):
        probs = probs.numpy()

    for i in range(len(images)):
        image = _get_image(images[i])
        label_idx = _get_label_idx(label_idxs[i])
        label = _get_label(label_idx, classes)
        imgfig = _build_image_figure(image, label, height, width)

        if isinstance(probs[i], Tensor):
            prob_dist = probs[i].numpy()
        else:
            prob_dist = probs[i]
        probfig = _build_prob_figure(prob_dist, label_idx, classes, height, width)
        imgfigs.append(imgfig)
        probfigs.append(probfig)

    show(grid([imgfigs, probfigs]))


def test_single():
    import imageio

    rgba = imageio.imread("./tf-logo.png")
    show_image(rgba)

    # rgb = imageio.imread("./tf-logo.jpg")
    # show_image(rgb)


def test_multiple():
    import tensorflow_datasets as tfds

    flowers, flowers_info = tfds.load("tf_flowers", data_dir="/data", with_info=True)
    classes = flowers_info.features["label"].names

    images = []
    labels = []
    for elem in flowers["train"].take(3):
        images.append(elem["image"])
        labels.append(elem["label"])
    # show_image(images[0])
    # show_image(images[0], labels[0])
    # show_image(images[0], labels[0], classes)
    # show_images(images)
    # show_images(images, labels)
    # show_images(images, labels, classes)
    # show_images(images, labels, classes, 192, 192)


def test_probs():
    import tensorflow_datasets as tfds
    import tensorflow as tf

    flowers, flowers_info = tfds.load("tf_flowers", data_dir="/data", with_info=True)
    classes = flowers_info.features["label"].names

    images = []
    labels = []
    for elem in flowers["train"].take(3):
        images.append(elem["image"])
        labels.append(elem["label"])
    logits = tf.nn.softmax(np.random.random((3, len(classes))))
    show_image_probs(images, logits, labels, classes)


if __name__ == "__main__":
    # test_single()
    # test_multiple()
    test_probs()
