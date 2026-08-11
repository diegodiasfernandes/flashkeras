from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *


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
        return (data.target_size[0], data.target_size[1], 3)

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

