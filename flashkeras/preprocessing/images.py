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
    img_shape: tuple[int, int] | Literal["auto"] = "auto",
    color_mode: Literal["rgb", "grayscale", "auto"] = "auto",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> NumpyArrayIterator:
    """Generates batches of augmented image data from NumPy arrays with automatic shape and color mode detection, and dynamic pixel rescaling.

    Args:
        x: NumPy array of input images.
        y: NumPy array of labels (optional).
        batch_size: Size of the batches.
        img_shape: Target image shape as (height, width) or "auto" to extract dimensions from input `x`.
        color_mode: Color mode of the images ("rgb", "grayscale", or "auto" to preserve native format).
        horizontal_flip: Whether to randomly flip images horizontally.
        rotation_range: Degree range for random rotations.
        zoom_range: Range for random zoom.
        brightness_range: Range for picking a brightness shift factor.
        fill_mode: Points outside the boundaries are filled according to the given mode.

    Returns ```img_batches```:
        A NumpyArrayIterator yielding batches of augmented images.
    """
    if img_shape == "auto":
        if x.ndim >= 3:
            img_shape = (x.shape[1], x.shape[2])
        else:
            raise ValueError("Input array 'x' must have at least 3 dimensions to automatically extract shape.")

    if color_mode == "rgb":
            x = prepro.convertNumpyNdArrayToRGB(x)
    elif color_mode == "grayscale":
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        x = prepro.convertNumpyNdArrayToGrayScale(x)
    else:
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
            
    x = prepro.resizeNpArray(x, img_shape[0], img_shape[1])

    rescale_val = 1.0 / 255 if np.max(x) > 1.0 else None

    data_gen = ImageDataGenerator(
        rescale=rescale_val,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    batches = data_gen.flow(x, y, batch_size, shuffle=True)
    return batches


def preprocess_images_from_nparray_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_split: float = 0.2,
    batch_size: int = 32,
    img_shape: tuple[int, int] | Literal["auto"] = "auto",
    color_mode: Literal["rgb", "grayscale", "auto"] = "auto",
    horizontal_flip: bool = False,
    rotation_range: int = 0,
    zoom_range: float = 0,
    brightness_range: tuple[float, float] | None = None,
    fill_mode: str = "nearest",
) -> tuple[NumpyArrayIterator, NumpyArrayIterator]:
    """Generates training and validation batches from NumPy arrays using a validation split, with automatic shape and color mode detection, and dynamic pixel rescaling.

    Args:
        x: NumPy array of input images.
        y: NumPy array of labels.
        test_split: Fraction of images reserved for validation (validation_split).
        batch_size: Size of the batches.
        img_shape: Target image shape as (height, width) or "auto" to extract dimensions from input `x`.
        color_mode: Color mode of the images ("rgb", "grayscale", or "auto" to preserve native format).
        horizontal_flip: Whether to randomly flip images horizontally.
        rotation_range: Degree range for random rotations.
        zoom_range: Range for random zoom.
        brightness_range: Range for picking a brightness shift factor.
        fill_mode: Points outside the boundaries are filled according to the given mode.

    Returns ```(train_batches, test_batches)```:
        A tuple containing training and validation NumpyArrayIterators.
    """
    if img_shape == "auto":
        if x.ndim >= 3:
            img_shape = (x.shape[1], x.shape[2])
        else:
            raise ValueError("Input array 'x' must have at least 3 dimensions to automatically extract shape.")

    if color_mode == "rgb":
            x = prepro.convertNumpyNdArrayToRGB(x)
    elif color_mode == "grayscale":
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        x = prepro.convertNumpyNdArrayToGrayScale(x)
    else:
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)

    x = prepro.resizeNpArray(x, img_shape[0], img_shape[1])
    y = prepro.ensureOneHotEncoding(y)

    rescale_val = 1.0 / 255 if np.max(x) > 1.0 else None

    data_gen = ImageDataGenerator(
        rescale=rescale_val,
        validation_split=test_split,
        horizontal_flip=horizontal_flip,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        brightness_range=brightness_range,
        fill_mode=fill_mode,
    )

    train_batches = data_gen.flow(x, y, batch_size, subset="training", shuffle=True)
    test_batches = data_gen.flow(x, y, batch_size, subset="validation", shuffle=True)

    return train_batches, test_batches


def stackImageDatasets(
        data_a: np.ndarray | Tuple[np.ndarray, np.ndarray], 
        data_b: np.ndarray | Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    
    """
    Stacks two image datasets vertically (concatenating along the first axis), 
    supporting both raw NumPy image arrays and `(images, labels)` dataset tuples.  

    `Flash Explanation:` *`Use this when you want to merge two batches or datasets of images 
    (with or without labels) into one continuous dataset, keeping everything aligned!`*  
    This function ensures that input shapes match (except for the first dimension), 
    then concatenates them row-wise. If labels are provided, they are also stacked 
    consistently with their corresponding images.  

    Parameters:
        data_a: np.ndarray | Tuple[np.ndarray, np.ndarray]  
            The first dataset. Can be:  
            - A NumPy array of images with shape `(n_samples, height, width, channels)`  
            - A tuple `(images, labels)`, where `images` and `labels` are both NumPy arrays.  

        data_b: np.ndarray | Tuple[np.ndarray, np.ndarray]  
            The second dataset. Must be of the same type and structure as `data_a`.  

    Returns:
        np.ndarray | Tuple[np.ndarray, np.ndarray]  
            - If both inputs are `np.ndarray`: returns a concatenated image array.  
            - If both inputs are `(images, labels)` tuples: returns a new tuple with 
            stacked images and labels.  

    Raises:
        ValueError:  
            - If image shapes (beyond the first dimension) do not match.  
            - If labels shapes (beyond the first dimension) do not match when using tuples.  
            - If tuples do not have exactly two elements `(images, labels)`.  

        TypeError:  
            - If the inputs are not both arrays or both `(images, labels)` tuples.  

    ---
    Examples:
    >>> import numpy as np
    >>> from my_module import stackImageDatasets

    #### Combine image batches (NumPy arrays only)
    >>> batch1 = np.random.rand(10, 64, 64, 3)   # 10 RGB images
    >>> batch2 = np.random.rand(5, 64, 64, 3)    # 5 RGB images
    >>> merged = stackImageDatasets(batch1, batch2)
    >>> print(merged.shape)
    (15, 64, 64, 3)

    ---
    #### Combine datasets with images and labels
    >>> images_a = np.random.rand(8, 32, 32, 1)         # 8 grayscale images
    >>> labels_a = np.random.randint(0, 2, (8, 1))      # binary labels
    >>> images_b = np.random.rand(4, 32, 32, 1)         # 4 grayscale images
    >>> labels_b = np.random.randint(0, 2, (4, 1))      # binary labels
    >>> merged_images, merged_labels = stackImageDatasets((images_a, labels_a), 
    ...                                                   (images_b, labels_b))
    >>> print(merged_images.shape, merged_labels.shape)
    (12, 32, 32, 1) (12, 1)
    """
    
    if isinstance(data_a, np.ndarray) and isinstance(data_b, np.ndarray):
        if data_a.shape[1:] != data_b.shape[1:]:
            raise ValueError(f"Shape mismatch: data_a has shape {data_a.shape} and data_b has shape {data_b.shape}. "
                                f"Both must have the same shape except for the first dimension.")

        return np.concatenate([data_a, data_b], axis=0)
    
    elif isinstance(data_a, tuple) and isinstance(data_b, tuple):
        if len(data_a) != 2 or len(data_b) != 2:
            raise ValueError("Both inputs should be tuples of length 2, (images, labels).")
        
        images_a, labels_a = data_a
        images_b, labels_b = data_b
        
        if images_a.shape[1:] != images_b.shape[1:]:
            raise ValueError(f"Shape mismatch in images: images_a has shape {images_a.shape} and images_b has shape {images_b.shape}. "
                                f"Both must have the same shape except for the first dimension.")
        
        if labels_a.shape[1:] != labels_b.shape[1:]:
            raise ValueError(f"Shape mismatch in labels: labels_a has shape {labels_a.shape} and labels_b has shape {labels_b.shape}. "
                                f"Both must have the same shape except for the first dimension.")
        
        merged_images = np.concatenate([images_a, images_b], axis=0)
        merged_labels = np.concatenate([labels_a, labels_b], axis=0)
        
        return (merged_images, merged_labels)
    
    else:
        raise TypeError("Inputs must be either both ndarrays or both tuples of (images, labels).")


def getInputShape(data: Union[np.ndarray, pd.DataFrame, DirectoryIterator, NumpyArrayIterator]) -> tuple:

    """
    Infers the input shape of a dataset (features-only) or iterator, supporting multiple data formats 
    (NumPy arrays, Pandas DataFrames, and Keras data iterators).  

    `Flash Explanation:` *`Use this when you want to automatically detect the input shape 
    of your dataset or generator! This can be placed as the input_shape parameter for all keras methods, such as Dense(input_shape=).`*  

    Parameters:
        data: np.ndarray | pd.DataFrame | DirectoryIterator | NumpyArrayIterator  
            The dataset or iterator whose input shape will be determined.  
            Supported inputs:  
            - **NumpyArrayIterator:** Returns the shape of the contained image batch 
            (excluding the batch dimension).  
            - **DirectoryIterator:** Returns `(height, width, 3)` assuming RGB images.  
            - **NumPy array (1D or 2D)** or **Pandas DataFrame:** Returns `(n_features,)`, 
            treating the data as tabular. Remember to drop the label column for this.
            - **NumPy array (3D or 4D):** Returns an image-like shape, either 
            `(height, width, 1)` for grayscale or `(height, width, 3)` for RGB.  

    Returns:
        tuple  
            A tuple describing the inferred input shape, excluding the batch dimension.  

    Raises:
        TypeError:  
            - If `data` is not one of the supported types.  

    ---  
    Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from keras.preprocessing.image import NumpyArrayIterator, DirectoryIterator

    #### From a NumpyArrayIterator
    >>> dummy_images = np.random.rand(10, 64, 64, 3)   # 10 RGB images
    >>> iterator = NumpyArrayIterator(dummy_images, np.arange(10), batch_size=2, shuffle=False)
    >>> shape = MyClass.getInputShape(iterator)
    >>> print(shape)
    (64, 64, 3)

    ---  
    #### From a DirectoryIterator
    >>> # Assuming a directory with RGB images resized to 128x128
    >>> dir_iter = DirectoryIterator("data/images", image_data_generator, target_size=(128,128))
    >>> shape = MyClass.getInputShape(dir_iter)
    >>> print(shape)
    (128, 128, 3)

    ---  
    #### From a NumPy array (tabular data)
    >>> arr = np.random.rand(100, 20)  # 100 samples, 20 features
    >>> shape = MyClass.getInputShape(arr)
    >>> print(shape)
    (20,)

    ---  
    #### From a Pandas DataFrame
    >>> df = pd.DataFrame(np.random.rand(50, 10))  # 50 samples, 10 features
    >>> shape = MyClass.getInputShape(df)
    >>> print(shape)
    (10,)

    ---  
    #### From a raw NumPy image
    >>> img = np.random.rand(28, 28)   # single grayscale image
    >>> shape = MyClass.getInputShape([img])
    >>> print(shape)
    (28, 28, 1)
    """

    
    if isinstance(data, NumpyArrayIterator):
        return data.x.shape[1:]

    if isinstance(data, DirectoryIterator):
        return data.image_shape

    if ((isinstance(data, np.ndarray) and data.ndim < 3) or isinstance(data, pd.DataFrame)):
        temp_data = data
        temp_data = pd.DataFrame(temp_data)
        return (temp_data.shape[1], )
    
    else:
        shape = data[0].shape
        if len(shape) == 2: 
            return (shape[0], shape[1], 1)
        else: 
            return (shape[0], shape[1], 3)   


def getImageShape(image: Union[np.ndarray, Image.Image, str]) -> Tuple[int, int]:
    '''Provide the image or path to the image and get its dimensions size i.e. (32, 32).
    '''

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return (image.shape[0], image.shape[1])
        elif image.ndim == 3:
            return (image.shape[0], image.shape[1])
    
    raise ValueError("The image must be one of the following types: ``np.ndarray``, ``Image.Image`` or a ``str`` representing the path.")


__all__ = [
    "preprocess_images_from_nparray",
    "preprocess_images_from_nparray_test_split",
    "stackImageDatasets",
    "getImageShape",
    "getInputShape"
]
