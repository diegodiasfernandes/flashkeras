from keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.preprocessing.image import DirectoryIterator # type: ignore
from utils.filesutils import *

class FlashDataGenerator:
    def __init__(self,         
                    img_size: int,
                    color_mode: str = "rgb",
                    horizontal_flip: bool = False,
                    rotation_range: int = 0,
                    zoom_range: float = 0,
                    brightness_range: tuple[float, float] | None = None,
                    fill_mode: str = "nearest"
                ) -> None:
    
        self.img_size: int = img_size
        self.color_mode: str = color_mode
        self.horizontal_flip: bool = horizontal_flip
        self.rotation_range: int = rotation_range
        self.zoom_range: float = zoom_range
        self.brightness_range: tuple[float, float] | None = brightness_range
        self.fill_mode: str = fill_mode

    def flow_all_classes_from_dir(
            self,
            path_to_main_dir: str, 
            ) -> DirectoryIterator | None:
        
        class_mode = None
        num_classes = len(count_directories_in_directory(path_to_main_dir))
        if num_classes == 2:
            class_mode = "binary"
        elif num_classes > 2:
            class_mode = "categorical"
        else:
            print("Invalid number of classes! Must be greater than 2.") 
            return None
        
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
            batch_size=32,
            shuffle=True,
        )

        return batches

    def flow_classes_from_dir(
            self,
            path_to_main_dir: str, 
            classes: list[str]
            ) -> DirectoryIterator | None:
        
        class_mode = None
        num_classes = len(classes)
        if num_classes == 2:
            class_mode = "binary"
        elif num_classes > 2:
            class_mode = "categorical"
        else:
            print("Invalid number of classes! Must be greater than 2.") 
            return None
        
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
            batch_size=32,
            shuffle=True,
        )

        return batches

    def flow_classes_from_dir_test_split(
            self,
            path_to_main_dir: str, 
            classes: list[str],
            test_split: float = 0.2
            ) -> tuple[DirectoryIterator, DirectoryIterator] | None:
        
        class_mode = None
        num_classes = len(classes)
        if num_classes == 2:
            class_mode = "binary"
        elif num_classes > 2:
            class_mode = "categorical"
        else:
            print("Invalid number of classes! Must be greater than 2.") 
            return None
        
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
            batch_size=32,
            shuffle=True,
            subset='training'
        )

        test_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=(self.img_size, self.img_size),
            class_mode=class_mode,
            classes=classes,
            batch_size=32,
            shuffle=True,
            subset='validation'
        )

        return train_batches, test_batches