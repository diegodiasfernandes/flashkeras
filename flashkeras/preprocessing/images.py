from __future__ import annotations

from typing import Literal

import numpy as np

from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *
from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as prepro


def preprocess_images_from_nparray(
    x: np.ndarray,
    y: np.ndarray | None = None,
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> NumpyArrayIterator:
    """Generates batches of augmented image data from NumPy arrays."""
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    if color_mode == "rgb":
        x = prepro.convertNumpyNdArrayToRGB(x)
    if color_mode == "grayscale":
        x = prepro.convertNumpyNdArrayToGrayScale(x)

    x = prepro.resizeNpArray(x, img_shape[0], img_shape[1])

    batches = data_gen.flow(x, y, batch_size, shuffle=True)
    return batches


def preprocess_images_from_nparray_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_split: float = 0.2,
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> tuple[NumpyArrayIterator, NumpyArrayIterator]:
    """Generates training and validation batches from NumPy arrays using a validation split."""
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=test_split,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    if color_mode == "rgb":
        x = prepro.convertNumpyNdArrayToRGB(x)
    if color_mode == "grayscale":
        x = prepro.convertNumpyNdArrayToGrayScale(x)

    x = prepro.resizeNpArray(x, img_shape[0], img_shape[1])
    y = prepro.ensureOneHotEncoding(y)

    train_batches = data_gen.flow(x, y, batch_size, subset="training", shuffle=True)
    test_batches = data_gen.flow(x, y, batch_size, subset="validation", shuffle=True)

    return train_batches, test_batches


__all__ = [
    "preprocess_images_from_nparray",
    "preprocess_images_from_nparray_test_split",
]
