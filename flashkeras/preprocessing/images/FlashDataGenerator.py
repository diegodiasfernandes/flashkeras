from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *
from flashkeras.utils.filesutils import count_directories_in_directory
from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as prepro

class FlashDataGenerator:
    """
    FlashDataGenerator

    A utility class for loading, preprocessing, and applying real-time augmentations 
    to image data from either directories or NumPy arrays, built on top of 
    `keras.preprocessing.image.ImageDataGenerator`.

    `Flash Explanation:` *`Use this class if you want an easy way to load and preprocess images for training or testing deep learning models!`*

    ---
    Parameters:
        img_shape (tuple[int, int]):
            Target size (height, width) for resizing all loaded images.

        color_mode (Literal["rgb", "grayscale"], default="rgb"):
            Color mode for loading images.
            - `"rgb"`: 3-channel color images.
            - `"grayscale"`: single-channel grayscale images.

        horizontal_flip (bool, default=False):
            Whether to randomly flip images horizontally during generation.

        rotation_range (int, default=0):
            Maximum degree range for random rotations. `0` means no rotation.

        zoom_range (float, default=0):
            Range for random zoom applied to images.

        brightness_range (tuple[float, float] | None, default=None):
            Range for random brightness adjustment. Example: `(0.8, 1.2)`.

        fill_mode (str, default="nearest"):
            Strategy for filling in newly created pixels after geometric transformations.  
            Options: `"nearest"`, `"reflect"`, `"wrap"`, `"constant"`.
    
    ---
    Examples:
    >>> generator = FlashDataGenerator(img_shape=(224, 224), horizontal_flip=True, rotation_range=15)
    
    #### Load training images from directory
    >>> train_batches = generator.flow_images_and_all_classes_from_dir("data/train", batch_size=32)
    
    #### Load training data from NumPy arrays
    >>> batches = generator.flow_images_from_nparray(x_train, y_train, batch_size=64)
    
    #### Automatically create train/validation split from a directory
    >>> train_batches, val_batches = generator.flow_images_and_all_classes_from_dir_test_split("data/train", test_split=0.2)
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
        
        `Flash Explanation:` *`If you have images on a directory and want to apply preprocessing use this!`*
        This function helps you load images from a directory and 
        apply a variety of image transformations like flipping, rotating, and zooming 
        to create augmented versions of your images. It's useful when you want to feed 
        images into a model for training, and you don't want to manually handle each 
        image. The function automatically handles resizing and rescaling, and can even 
        shuffle the images or leave them in order depending on your preference.

        Parameters:
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
        
            - `x` is a numpy array containing a batch of images with shape `(batch_size, target_size, channels)`, with the images rescaled and augmented according to the settings.
            - `y` is `None`, since no labels are returned (useful for tasks like unsupervised learning or inference).
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

    def flow_images_from_nparray(
        self, 
        x: np.ndarray, 
        y: np.ndarray | None = None,
        batch_size: int = 32
        ) -> NumpyArrayIterator:
        
        """
        Generates batches of augmented image data from NumPy arrays.

        `Flash Explanation:` *`If you have images as NumPy Array and want to apply preprocessing use this!`*
        This function takes in raw image data in the form of NumPy arrays 
        and returns an iterator that yields batches of images (and optionally labels), 
        applying a variety of augmentations such as flipping, rotating, zooming, and brightness 
        adjustments. It also resizes and rescales the input images automatically. This is 
        especially useful when your images are already loaded into memory and you want to 
        train a model without reading them from disk.

        Parameters:
            x: np.ndarray
                
                NumPy array of input images. These can be grayscale or RGB images, and they will be 
                automatically converted and resized based on the model's expected input shape and 
                color mode.

            y: np.ndarray or None, optional, default: None
                
                NumPy array of labels corresponding to the input images. If `None`, the iterator will 
                yield batches of images only, without labels. This can be useful for inference or 
                unsupervised tasks.

        batch_size: int, optional, default: 32
            Number of samples per batch to yield.

        Returns:
        A `NumpyArrayIterator` object that yields batches of augmented image data.
        Each batch is a tuple `(x_batch, y_batch)` where:
            - `x_batch` is a NumPy array of images with shape `(batch_size, *img_shape, channels)`, 
            rescaled to [0, 1] and augmented as specified.
            - `y_batch` is a NumPy array of labels, or `None` if no labels were provided.
        """
        
        data_gen = ImageDataGenerator(
                    rescale=1./255,
                    horizontal_flip=self.horizontal_flip,
                    rotation_range=self.rotation_range,
                    zoom_range=self.zoom_range,
                    brightness_range=self.brightness_range,
                    fill_mode=self.fill_mode
                )
        
        if self.color_mode == 'rgb':
            x = prepro.convertNumpyNdArrayToRGB(x)
        if self.color_mode == 'grayscale':
            x = prepro.convertNumpyNdArrayToGrayScale(x)

        x = prepro.resizeNpArray(x, self.img_shape[0], self.img_shape[1])
        
        batches = data_gen.flow(x, y, batch_size, shuffle=True)

        return batches
    
    def flow_images_from_nparray_test_split(
            self, 
            x: np.ndarray, 
            y: np.ndarray,
            test_split: float = 0.2,
            batch_size: int = 32
        ) -> tuple[NumpyArrayIterator, NumpyArrayIterator]:
        
        """
        Generates training and validation batches of augmented image data from NumPy arrays, with automatic test split.

        `Flash Explanation:` *`If you have images as NumPy Array and want to apply preprocessing and test_split use this!`*
        This function takes arrays of images and their corresponding class labels, 
        applies preprocessing (such as resizing, color conversion, and normalization), and returns 
        two iterators: one for training and one for validation. It uses Keras's internal splitting 
        mechanism (`validation_split`) to divide the dataset and applies real-time data augmentation 
        to improve generalization during training.

        Parameters:
            x: np.ndarray  
                NumPy array of input images. The images will be resized and converted according to 
                the model's expected input shape and color mode (`rgb` or `grayscale`).

            y: np.ndarray  
                NumPy array of class labels corresponding to each image in `x`. These labels will be 
                automatically one-hot encoded to match the output format expected by most classification models.

            test_split: float, optional, default: 0.2  
                Fraction of the data to be used as validation. The value should be between 0 and 1.  
                For example, `0.2` means 80% training and 20% validation.

            batch_size: int, optional, default: 32  
                Number of samples per batch to yield.

        Returns:
        A tuple of two `NumpyArrayIterator` objects:
            - The first iterator yields batches of training data.
            - The second iterator yields batches of validation (test) data.

        Each batch is a tuple `(x_batch, y_batch)` where:
            - `x_batch` is a NumPy array of shape `(batch_size, *img_shape, channels)`, rescaled to [0, 1] and augmented.
            - `y_batch` is a one-hot encoded array of class labels corresponding to the images.
        """
        
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
            x = prepro.convertNumpyNdArrayToRGB(x)
        if self.color_mode == 'grayscale':
            x = prepro.convertNumpyNdArrayToGrayScale(x)
    
        x = prepro.resizeNpArray(x, self.img_shape[0], self.img_shape[1])
        
        y = prepro.ensureOneHotEncoding(y)
        
        train_batches = data_gen.flow(x, y, batch_size, subset='training', shuffle=True)

        test_batches = data_gen.flow(x, y, batch_size, subset='validation', shuffle=True)

        return (train_batches, test_batches) 

    def flow_images_and_all_classes_from_dir(
            self,
            path_to_main_dir: str, 
            batch_size: int = 32
            ) -> DirectoryIterator | None:
        
        """
        Generates batches of augmented image data directly from a directory structure.

        `Flash Explanation:` *`Use this when your images are organized in folders per class and you want real-time augmentation!`*  
        This function reads images from a main directory where each subdirectory represents a class. 
        It applies real-time data augmentation and preprocessing to the images and yields batches suitable for training or evaluation.

        This is particularly useful for large datasets that don't fit into memory, as images are loaded and processed in real-time during training.

        Parameters:
            path_to_main_dir: str  
                Path to the main directory containing one subdirectory per class. Each subdirectory should contain images belonging to that class.  
                Example structure:  
                ```
                path_to_main_dir/
                    class_1/
                        img001.jpg
                        img002.jpg
                    class_2/
                        img003.jpg
                        img004.jpg
                ```

            batch_size: int, optional, default: 32  
                Number of images to yield per batch.

        Returns:
            DirectoryIterator or None  
                An iterator over the dataset that yields batches `(x_batch, y_batch)`, where:  
                    - `x_batch` is a batch of image data, rescaled to `[0, 1]` and augmented in real-time.  
                    - `y_batch` is a batch of class labels in either one-hot or categorical format depending on the detected class mode (`'categorical'`, `'binary'`, `'sparse'`, etc.).  
                Returns `None` if the class mode could not be determined or directory is invalid.

        Details:
            - Uses `ImageDataGenerator.flow_from_directory()` under the hood.
            - Applies the following augmentations (configurable via class attributes):
                - Horizontal flip
                - Rotation
                - Zoom
                - Brightness adjustment
                - Fill mode for new pixels after transformation
            - Automatically infers class mode (`categorical`, `binary`, etc.) from directory structure via `_getClassMode()` helper.
        """
        
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
    
    def flow_images_and_all_classes_from_dir_test_split(
            self,
            path_to_main_dir: str,
            test_split: float = 0.2, 
            batch_size: int = 32
            ) -> tuple[DirectoryIterator, DirectoryIterator] | None:

        """
        Generates training and validation batches of augmented image data from a directory, using automatic test split.

        `Flash Explanation:` *`Use this when your images are in folders by class and you want to split them automatically into training and validation sets with augmentation!`*  
        This function reads images from a directory where each subfolder represents a class. It uses Keras's internal `validation_split` mechanism to automatically separate the data into training and validation sets. Augmentations and preprocessing are applied in real time.

        Parameters:
            path_to_main_dir: str  
                Path to the main directory containing class-named subdirectories with images.  
                Example structure:  
                ```
                path_to_main_dir/
                    class_1/
                        img001.jpg
                        img002.jpg
                    class_2/
                        img003.jpg
                        img004.jpg
                ```

            test_split: float, optional, default: 0.2  
                Fraction of the data to reserve for validation. Must be a float between 0 and 1.  
                For example, `0.2` means 80% training and 20% validation.

            batch_size: int, optional, default: 32  
                Number of samples per batch.

        Returns:
            tuple of DirectoryIterator or None  
                A tuple `(train_batches, test_batches)` where each element is a `DirectoryIterator` that yields batches in the format `(x_batch, y_batch)`, with:
                - `x_batch`: A batch of preprocessed and augmented image data (rescaled to `[0, 1]`).
                - `y_batch`: Corresponding class labels in a format based on the inferred `class_mode` (`'categorical'`, `'binary'`, etc.).

                Returns `None` if class mode inference fails or the directory is invalid.

        Details:
            - Uses `ImageDataGenerator.flow_from_directory()` with `subset='training'` and `subset='validation'`.
            - Enables real-time data augmentation during both training and validation phases.
            - Applies the following augmentations (controlled by class attributes):
                - Horizontal flipping
                - Rotation
                - Zoom
                - Brightness adjustment
                - Custom fill mode for transformations
            - Class mode (`categorical`, `binary`, or `sparse`) is determined automatically via `_getClassMode()` helper.
        """

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

    def flow_images_and_classes_from_dir(
            self,
            path_to_main_dir: str, 
            classes: list[str], 
            batch_size: int = 32
            ) -> DirectoryIterator | None:
        
        """
        Generates training batches of augmented image data from a directory, restricted to specific classes.

        `Flash Explanation:` *`Use this when your images are organized in folders per class and you want real-time augmentation for some of the classes!`*  
        This function reads images from a directory and restricts the loading to the user-specified classes. Each subfolder must represent a class. Real-time augmentations and preprocessing are applied during batch generation.

        Parameters:
            path_to_main_dir: str  
                Path to the main directory containing subdirectories for each class.  
                Example structure:  
                ```
                path_to_main_dir/
                    class_1/
                        img001.jpg
                        img002.jpg
                    class_2/
                        img003.jpg
                        img004.jpg
                ```

            classes: list[str]  
                List of class subdirectory names to include.  
                Example: `["class_1", "class_2"]`.  
                Only these classes will be considered, even if more exist in the directory.

            batch_size: int, optional, default: 32  
                Number of samples per batch.

        Returns:
            DirectoryIterator or None  
                A `DirectoryIterator` that yields batches in the format `(x_batch, y_batch)`, with:
                - `x_batch`: Preprocessed and augmented images, rescaled to `[0, 1]`.
                - `y_batch`: Class labels inferred from the `classes` list.

                Returns `None` if class mode inference fails or directory is invalid.

        Details:
            - Uses `ImageDataGenerator.flow_from_directory()` with the `classes` argument to restrict data loading.
            - Augmentations applied include:
                - Horizontal flipping
                - Rotation
                - Zoom
                - Brightness adjustment
                - Custom fill mode for transformations
            - `class_mode` is automatically determined by `_getClassMode()`.
        """
        
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

    def flow_images_and_classes_from_dir_test_split(
            self,
            path_to_main_dir: str, 
            classes: list[str],
            test_split: float = 0.2, 
            batch_size: int = 32
            ) -> tuple[DirectoryIterator, DirectoryIterator] | None:
        
        """
        Generates training and validation batches of augmented image data from a directory, restricted to specific classes, using automatic test split.

        `Flash Explanation:` *`Use this when your images are organized in folders per class and you want real-time augmentation for some of the classes! Also splitting the data into train and test!`*  
        This function reads images from a directory, restricted to the given list of classes. It uses Keras' `validation_split` mechanism to separate data into training and validation sets. Augmentations and preprocessing are applied dynamically.

        Parameters:
            path_to_main_dir: str  
                Path to the main directory containing subdirectories for each class.  
                Example structure:  
                ```
                path_to_main_dir/
                    class_1/
                        img001.jpg
                        img002.jpg
                    class_2/
                        img003.jpg
                        img004.jpg
                ```

            classes: list[str]  
                List of class subdirectory names to include.  
                Example: `["class_1", "class_2"]`.

            test_split: float, optional, default: 0.2  
                Fraction of the data to reserve for validation. Must be between `0` and `1`.  
                For example, `0.2` means 80% training and 20% validation.

            batch_size: int, optional, default: 32  
                Number of samples per batch.

        Returns:
            tuple of DirectoryIterator or None  
                A tuple `(train_batches, test_batches)` where each element is a `DirectoryIterator` yielding `(x_batch, y_batch)`:
                - `x_batch`: A batch of preprocessed and augmented image data (rescaled to `[0, 1]`).
                - `y_batch`: Corresponding class labels restricted to the specified `classes`.

                Returns `None` if class mode inference fails or directory is invalid.

        Details:
            - Uses `ImageDataGenerator.flow_from_directory()` with:
                - `subset='training'` for training batches
                - `subset='validation'` for validation batches
            - Augmentations applied include:
                - Horizontal flipping
                - Rotation
                - Zoom
                - Brightness adjustment
                - Custom fill mode for transformations
            - `class_mode` is automatically determined by `_getClassMode()`.
        """
        
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