from flashkeras.preprocessing.images import (
    convertNumpyNdArrayToGrayScale,
    convertNumpyNdArrayToRGB,
    getImageShape,
    getInputShape,
    normalize_pixels,
    resizeNpArray,
    stackImageDatasets,
)
from flashkeras.preprocessing.tabular.conversion import datasetPandasToNumpyNDArray
from flashkeras.preprocessing.tabular.encoding import (
    ensureOneHotEncoding,
    labelDecoder,
    labelEncoder,
)
from flashkeras.preprocessing.tabular.multiple_df_operations import stackDataFrames
from flashkeras.preprocessing.tabular.scaling import minMaxScaleRevert, minMaxScaler
from flashkeras.preprocessing.tabular.splitting import train_test_split


class FlashPreProcessing:
    """
    FlashPreProcessing

    A comprehensive utility class for preprocessing, transforming, and preparing datasets, 
    images, and labels for machine learning workflows. It provides methods for splitting, 
    stacking, normalizing, encoding, resizing, and format conversions, supporting both 
    tabular and image data.

    `Flash Explanation:` *`Use this class if you want an all-in-one preprocessing toolkit 
    for preparing data before feeding it into deep learning or machine learning models!`*

    ---
    Main Functionalities:
    
    • **Data Splitting**:
        - `train_test_split`: Split arrays into training and testing sets.
    • **Data Stacking**:
        - `stackDataFrames`: Vertically stack DataFrames, Series, or NumPy arrays.
        - `stackImageDatasets`: Merge image datasets or (images, labels) tuples.
    • **Shape Inference**:
        - `getInputShape`: Automatically detect the input shape from arrays, DataFrames, or Keras iterators.
        - `getImageShape`: Get the dimensions (H, W) of an image from a path, PIL object, or NumPy array.
    • **Label Handling**:
        - `ensureOneHotEncoding`: Guarantee labels are in one-hot format.
        - `labelEncoder`: Encode categorical labels into integers.
        - `labelDecoder`: Decode integer labels back into categories.
    • **Format Conversions**:
        - `datasetPandasToNumpyNDArray`: Convert DataFrames or Series to NumPy arrays.
        - `convertNumpyNdArrayToGrayScale`: Convert RGB image batches to grayscale.
        - `convertNumpyNdArrayToRGB`: Convert grayscale images to RGB.
    • **Image Preprocessing**:
        - `resizeNpArray`: Resize image arrays to a new target shape.
        - `normalize_pixels`: Normalize image pixels to range [0, 1].
    • **Feature Scaling**:
        - `minMaxScaler`: Apply Min-Max normalization to features.
        - `minMaxScaleRevert`: Revert scaled features back to original values.

    ---
    Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flash_module import FlashPreProcessing

    #### Split dataset
    >>> X = np.arange(10).reshape((5, 2))
    >>> y = np.array([0, 1, 0, 1, 0])
    >>> x_train, x_test, y_train, y_test = FlashPreProcessing.train_test_split(X, y, test_split=0.2, random_state=42)

    #### Stack two DataFrames
    >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    >>> stacked = FlashPreProcessing.stackDataFrames(df1, df2)

    #### Normalize images
    >>> images = np.random.randint(0, 255, (10, 28, 28, 1), dtype=np.uint8)
    >>> normalized = FlashPreProcessing.normalize_pixels(images)

    #### One-hot encode labels
    >>> labels = np.array([0, 1, 2, 1])
    >>> onehot = FlashPreProcessing.ensureOneHotEncoding(labels)
    """

    train_test_split = staticmethod(train_test_split)
    stackDataFrames = staticmethod(stackDataFrames)
    stackImageDatasets = staticmethod(stackImageDatasets)
    getInputShape = staticmethod(getInputShape)
    getImageShape = staticmethod(getImageShape)
    ensureOneHotEncoding = staticmethod(ensureOneHotEncoding)
    datasetPandasToNumpyNDArray = staticmethod(datasetPandasToNumpyNDArray)
    convertNumpyNdArrayToGrayScale = staticmethod(convertNumpyNdArrayToGrayScale)
    resizeNpArray = staticmethod(resizeNpArray)
    convertNumpyNdArrayToRGB = staticmethod(convertNumpyNdArrayToRGB)
    normalize_pixels = staticmethod(normalize_pixels)
    minMaxScaler = staticmethod(minMaxScaler)
    minMaxScaleRevert = staticmethod(minMaxScaleRevert)
    labelEncoder = staticmethod(labelEncoder)
    labelDecoder = staticmethod(labelDecoder)
