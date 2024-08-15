from keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.preprocessing.image import DirectoryIterator # type: ignore
from utils.filesutils import *
from typing import overload, Union, Tuple, Literal

COLOR_MODE = Literal["rgb", "grayscale"]

class FlashDataGenerator:
    def __init__(self,         
                    img_size: int,
                    color_mode: COLOR_MODE = "rgb",
                    horizontal_flip: bool = False,
                    rotation_range: int = 0,
                    zoom_range: float = 0,
                    brightness_range: tuple[float, float] | None = None,
                    fill_mode: str = "nearest"
                ) -> None:
    
        self.img_size = img_size
        self.color_mode = color_mode
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.fill_mode = fill_mode

    def _getClassMode(self, path_or_class_list: str | list[str]):
        num_classes: int = 0
        if type(path_or_class_list) == str:
            num_classes = count_directories_in_directory(path_or_class_list)
        else:
            num_classes = len(path_or_class_list)

        class_mode: str = ""
        if num_classes == 2:
            class_mode = "binary"
        elif num_classes > 2 or num_classes == 1:
            class_mode = "categorical"
        else:
            raise ValueError ("Invalid number of classes!.")

        return class_mode

    def flow_all_classes_from_dir(
            self,
            path_to_main_dir: str, 
            batch_size: int = 32
            ) -> DirectoryIterator | None:
        
        class_mode = self._getClassMode(path_to_main_dir)
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )

        batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=True,
        )

        return batches
    
    def flow_all_classes_from_dir_test_split(
            self,
            path_to_main_dir: str,
            test_split: float = 0.2, 
            batch_size: int = 32
            ) -> tuple[DirectoryIterator, DirectoryIterator] | None:

        class_mode = self._getClassMode(path_to_main_dir)

        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=test_split,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )

        train_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=True,
            subset='training'
        )

        test_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=True,
            subset='validation'
        )

        return train_batches, test_batches

    def flow_classes_from_dir(
            self,
            path_to_main_dir: str, 
            classes: list[str], 
            batch_size: int = 32
            ) -> DirectoryIterator | None:
        
        class_mode = self._getClassMode(classes)
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )

        batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=True,
        )

        return batches

    def flow_classes_from_dir_test_split(
            self,
            path_to_main_dir: str, 
            classes: list[str],
            test_split: float = 0.2, 
            batch_size: int = 32
            ) -> tuple[DirectoryIterator, DirectoryIterator] | None:
        
        class_mode: str = self._getClassMode(classes)
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=test_split,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )

        train_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=True,
            subset='training'
        )

        test_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=True,
            subset='validation'
        )

        return train_batches, test_batches