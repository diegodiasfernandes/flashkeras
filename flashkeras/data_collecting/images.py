from __future__ import annotations

from typing import Literal

from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.filesutils import count_directories_in_directory
from flashkeras.utils.typehints import DirectoryIterator


def _get_class_mode(path_or_class_list: str | list[str]) -> str:
    """Infers the Keras class mode from a directory path or a class list."""
    num_classes: int = 0
    if isinstance(path_or_class_list, str):
        num_classes = count_directories_in_directory(path_or_class_list)
    else:
        num_classes = len(path_or_class_list)

    if num_classes == 2:
        return "binary"
    if num_classes > 2 or num_classes == 1:
        return "categorical"
    raise ValueError("Invalid number of classes!.")


def load_from_directory_and_preprocess(
    directory_path: str,
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> DirectoryIterator:
    """Loads batches of images directly from a directory without keeping them in memory."""
    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    return data_gen.flow_from_directory(
        directory_path,
        color_mode=color_mode,
        target_size=img_shape,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )


def load_all_classes_from_directory_and_preprocess(
    path_to_main_dir: str,
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> DirectoryIterator | None:
    """Loads all classes from a directory tree as labeled image batches."""
    class_mode = _get_class_mode(path_to_main_dir)

    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    return data_gen.flow_from_directory(
        path_to_main_dir,
        color_mode=color_mode,
        target_size=img_shape,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
    )


def load_all_classes_from_directory_and_preprocess_test_split(
    path_to_main_dir: str,
    test_split: float = 0.2,
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> tuple[DirectoryIterator, DirectoryIterator] | None:
    """Loads all classes from a directory tree and splits train/validation automatically."""
    class_mode = _get_class_mode(path_to_main_dir)

    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=test_split,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    train_batches = data_gen.flow_from_directory(
        path_to_main_dir,
        color_mode=color_mode,
        target_size=img_shape,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
        subset="training",
    )

    test_batches = data_gen.flow_from_directory(
        path_to_main_dir,
        color_mode=color_mode,
        target_size=img_shape,
        class_mode=class_mode,
        batch_size=batch_size,
        shuffle=True,
        subset="validation",
    )

    return train_batches, test_batches


def load_classes_from_directory_and_preprocess(
    path_to_main_dir: str,
    classes: list[str],
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> DirectoryIterator | None:
    """Loads only selected classes from a directory tree as image batches."""
    class_mode = _get_class_mode(classes)

    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    return data_gen.flow_from_directory(
        path_to_main_dir,
        color_mode=color_mode,
        target_size=img_shape,
        class_mode=class_mode,
        classes=classes,
        batch_size=batch_size,
        shuffle=True,
    )


def load_classes_from_directory_and_preprocess_test_split(
    path_to_main_dir: str,
    classes: list[str],
    test_split: float = 0.2,
    batch_size: int = 32,
    img_shape: tuple[int, int] = (224, 224),
    color_mode: Literal["rgb", "grayscale"] = "rgb",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> tuple[DirectoryIterator, DirectoryIterator] | None:
    """Loads selected classes and splits them into training and validation batches."""
    class_mode: str = _get_class_mode(classes)

    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=test_split,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    train_batches = data_gen.flow_from_directory(
        path_to_main_dir,
        color_mode=color_mode,
        target_size=img_shape,
        class_mode=class_mode,
        classes=classes,
        batch_size=batch_size,
        shuffle=True,
        subset="training",
    )

    test_batches = data_gen.flow_from_directory(
        path_to_main_dir,
        color_mode=color_mode,
        target_size=img_shape,
        class_mode=class_mode,
        classes=classes,
        batch_size=batch_size,
        shuffle=True,
        subset="validation",
    )

    return train_batches, test_batches


__all__ = [
    '_get_class_mode',
    'load_from_directory_and_preprocess',
    'load_all_classes_from_directory_and_preprocess',
    'load_all_classes_from_directory_and_preprocess_test_split',
    'load_classes_from_directory_and_preprocess',
    'load_classes_from_directory_and_preprocess_test_split',
]
