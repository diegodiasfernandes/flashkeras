from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *
from flashkeras.utils.filesutils import count_directories_in_directory
from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as prepro

class FlashDataGenerator:
    """ Loads and preprocess images from directories, numpy arrays
    
    """
    def __init__(self,         
                    img_shape: tuple[int, int],
                    color_mode: Literal["rgb", "grayscale"] = "rgb",
                    horizontal_flip: bool = False,
                    rotation_range: int = 0,
                    zoom_range: float = 0,
                    brightness_range: tuple[float, float] | None = None,
                    fill_mode: str = "nearest"
                ) -> None:
    
        self.img_shape: tuple[int, int] = img_shape
        self.color_mode: Literal["rgb", "grayscale"] = color_mode
        self.horizontal_flip: bool = horizontal_flip
        self.rotation_range: int = rotation_range
        self.zoom_range: float = zoom_range
        self.brightness_range: tuple[float, float] | None = brightness_range
        self.fill_mode: str = fill_mode

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

    def flow_images_from_directory(
        self,
        directory_path: str,
        batch_size: int = 32
        ) -> DirectoryIterator:
        
        """
        Generates batches of augmented image data from a directory.
        
        *`Flash Explanation:`* This function helps you load images from a directory and 
        apply a variety of image transformations like flipping, rotating, and zooming 
        to create augmented versions of your images. It's useful when you want to feed 
        images into a model for training, and you don't want to manually handle each 
        image. The function automatically handles resizing and rescaling, and can even 
        shuffle the images or leave them in order depending on your preference.

        Args:
        directory_path: string, path to the target directory containing image files.
            The directory should contain images that will be augmented using the specified
            parameters. Images will be rescaled, rotated, flipped, and zoomed based on
            the settings in the `ImageDataGenerator`. The directory can also contain
            subdirectories, which will be used to infer class labels for the images.
        
        batch_size: int, optional, default: 32.
            The size of the batches of data to generate. The function will yield batches
            of this size from the directory.

        Returns:
        A `DirectoryIterator` object that yields batches of augmented image data.
        Each batch is a tuple `(x, y)` where:
            - `x` is a numpy array containing a batch of images with shape
            `(batch_size, *target_size, channels)`, with the images rescaled
            and augmented according to the settings.
            - `y` is `None`, since no labels are returned (useful for tasks like
            unsupervised learning or inference).
        """
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )

        image_iterator = data_gen.flow_from_directory(
            directory_path,
            color_mode=self.color_mode,
            target_size=self.img_shape,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False
        )

        return image_iterator

    def flow_classes_from_nparray(self, 
                                  x: np.ndarray, 
                                  y: np.ndarray | None = None,
                                  batch_size: int = 32
                                  ) -> NumpyArrayIterator:
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )
        
        if self.color_mode == 'rgb':
            x = prepro.convertNdArrayToRGB(x)
        if self.color_mode == 'grayscale':
            x = prepro.convertNdArrayToGrayScale(x)

        x = prepro.resizeNpArray(x, self.img_shape[0], self.img_shape[1])
        
        batches = data_gen.flow(x, y, batch_size, shuffle=True)

        return batches
    
    def flow_classes_from_nparray_test_split(self, 
                                  x: np.ndarray, 
                                  y: np.ndarray,
                                  test_split: float = 0.2,
                                  batch_size: int = 32
                                  ) -> tuple[NumpyArrayIterator, NumpyArrayIterator]:
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=test_split,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )
        
        if self.color_mode == 'rgb':
            x = prepro.convertNdArrayToRGB(x)
        if self.color_mode == 'grayscale':
            x = prepro.convertNdArrayToGrayScale(x)
    
        x = prepro.resizeNpArray(x, self.img_shape[0], self.img_shape[1])
        
        y = prepro.ensureOneHotEncoding(y)
        
        train_batches = data_gen.flow(x, y, batch_size, subset='training', shuffle=True)

        test_batches = data_gen.flow(x, y, batch_size, subset='validation', shuffle=True)

        return (train_batches, test_batches) 

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
            target_size=self.img_shape,
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
            target_size=self.img_shape,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=True,
            subset='training'
        )

        test_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=self.img_shape,
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
            target_size=self.img_shape,
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
            target_size=self.img_shape,
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=True,
            subset='training'
        )

        test_batches = data_gen.flow_from_directory(
            path_to_main_dir,
            color_mode=self.color_mode,
            target_size=self.img_shape,
            class_mode=class_mode,
            classes=classes,
            batch_size=batch_size,
            shuffle=True,
            subset='validation'
        )

        return train_batches, test_batches