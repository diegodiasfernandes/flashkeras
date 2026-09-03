from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

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

    @staticmethod
    def train_test_split(*arrays: Any, 
                         test_split: float | None = None,
                         random_state: int | None = None,
                         ) -> list:
        """Split arrays or matrices into random train and test subsets.
        
        `Flash Explanation:` *`Use this if you want split your data on train-test split!`*

        Quick utility that wraps input validation,
        ``next(ShuffleSplit().split(X, y))``, and application to input data
        into a single call for splitting (and optionally subsampling) data into a
        one-liner.

        Read more in the :ref:`User Guide <cross_validation>`.

        Parameters
        ----------
        *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays, scipy-sparse
            matrices or pandas dataframes.

        test_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. If ``train_size`` is also None, it will
            be set to 0.25.

        train_size : float or int, default=None
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.

        random_state : int, RandomState instance or None, default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        stratify : array-like, default=None
            If not None, data is split in a stratified fashion, using this as
            the class labels.
            Read more in the :ref:`User Guide <stratification>`.

        Returns
        -------
        splitting : list, length=2 * len(arrays)
            List containing train-test split of inputs.

            .. versionadded:: 0.16
                If the input is sparse, the output will be a
                ``scipy.sparse.csr_matrix``. Else, output type is the same as the
                input type.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = np.arange(10).reshape((5, 2)), range(5)
        >>> X
        array([[0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9]])
        >>> list(y)
        [0, 1, 2, 3, 4]

        >>> x_train, x_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.33, random_state=42)
        ...
        >>> x_train
        array([[4, 5],
            [0, 1],
            [6, 7]])
        >>> y_train
        [2, 0, 3]
        >>> x_test
        array([[2, 3],
            [8, 9]])
        >>> y_test
        [1, 4]

        >>> train_test_split(y, shuffle=False)
        [[0, 1, 2], [3, 4]]
        """
        return train_test_split(*arrays, test_size=test_split, random_state=random_state)

    @staticmethod
    def stackDataFrames(
            matrix_a: pd.DataFrame | pd.Series | np.ndarray, 
            matrix_b: pd.DataFrame | pd.Series | np.ndarray
        ) -> pd.DataFrame | pd.Series | np.ndarray:
        
        """
        Stacks two data structures (DataFrames, Series, or NumPy arrays) vertically 
        while preserving their type and structure.
        
        `Flash Explanation:` *`Use this when you want to combine two DataFrames, Series, or NumPy arrays into one, making sure they stay consistent in format!`*  
        This function converts both inputs into 2D arrays, verifies they have the same 
        number of columns, stacks them row-wise, and returns the result in the 
        appropriate format (DataFrame, Series, or NumPy array) based on the inputs.

        Parameters:
            matrix_a: pd.DataFrame | pd.Series | np.ndarray  
                The first data structure to be stacked. Can be a pandas DataFrame, 
                Series, or NumPy array.  
            matrix_b: pd.DataFrame | pd.Series | np.ndarray  
                The second data structure to be stacked. Must be of the same type 
                as `matrix_a` and have the same number of columns (if 2D).

        Returns:
            pd.DataFrame | pd.Series | np.ndarray  
                - If both inputs are `pd.DataFrame`: returns a new DataFrame with 
                combined rows and original column names.  
                - If both inputs are `pd.Series`: returns a new Series with combined 
                values and the same name as the original Series.  
                - If both inputs are `np.ndarray`: returns a stacked NumPy array. 
                1D arrays are flattened back after stacking.  
                - If inputs differ in type: returns a NumPy array.  

        Raises:
            ValueError:  
                - If the inputs are not `pd.DataFrame`, `pd.Series`, or `np.ndarray`.  
                - If the number of columns in the two inputs do not match.  

        ---
        Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from my_module import stackDataFrames

        #### Combine DataFrames
        >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        >>> result = stackDataFrames(df1, df2)
        >>> print(result)
           a  b
        0  1  3
        1  2  4
        2  5  7
        3  6  8

        ---
        #### Combine Series
        >>> s1 = pd.Series([1, 2, 3], name="x")
        >>> s2 = pd.Series([4, 5], name="x")
        >>> result = stackDataFrames(s1, s2)
        >>> print(result)
        0    1
        1    2
        2    3
        3    4
        4    5
        Name: x, dtype: int64

        ---
        #### Combine NumPy arrays
        >>> a1 = np.array([[1, 2], [3, 4]])
        >>> a2 = np.array([[5, 6]])
        >>> result = stackDataFrames(a1, a2)
        >>> print(result)
        [[1 2]
         [3 4]
         [5 6]]
        """

        
        def to_2d_array(data):
            """Convert a supported tabular value to a two-dimensional array.

            `Flash Explanation:` *`Use this internal helper to normalize shapes before stacking rows.`*
            """
            if isinstance(data, pd.DataFrame):
                return data.values
            elif isinstance(data, pd.Series):
                return data.values.reshape(-1, 1)
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    return data.reshape(-1, 1)
                return data
            else:
                raise ValueError("Inputs must be either pd.DataFrame, pd.Series, or np.ndarray")

        matrix_a_2d = to_2d_array(matrix_a)
        matrix_b_2d = to_2d_array(matrix_b)

        if matrix_a_2d.shape[1] != matrix_b_2d.shape[1]:
            raise ValueError(f"Shape mismatch: matrix_a has shape {matrix_a_2d.shape} and matrix_b has shape {matrix_b_2d.shape}. Both must have the same number of columns.")

        stacked = np.vstack((matrix_a_2d, matrix_b_2d))

        if isinstance(matrix_a, pd.DataFrame) and isinstance(matrix_b, pd.DataFrame):
            return pd.DataFrame(stacked, columns=matrix_a.columns)
        
        elif isinstance(matrix_a, pd.Series) and isinstance(matrix_b, pd.Series):
            return pd.Series(stacked.flatten(), name=matrix_a.name)
        
        elif isinstance(matrix_a, np.ndarray) and isinstance(matrix_b, np.ndarray):
            if matrix_a.ndim == 1 and matrix_b.ndim == 1:
                return stacked.flatten()
            return stacked 
        
        return stacked

    @overload
    @staticmethod
    def stackImageDatasets(data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray: """Array overload. `Flash Explanation:` *`Use this signature for two image arrays.`*"""
    
    @overload
    @staticmethod
    def stackImageDatasets(data_a: Tuple[np.ndarray, np.ndarray], data_b: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: """Tuple overload. `Flash Explanation:` *`Use this signature for image and label pairs.`*"""
    
    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def getImageShape(image: Union[np.ndarray, Image.Image, str]) -> Tuple[int, int]:
        '''Provide the image or path to the image and get its dimensions size i.e. (32, 32).

        `Flash Explanation:` *`Use this to obtain height and width before image preprocessing.`*
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

    @staticmethod
    def ensureOneHotEncoding(
            y: Union[np.ndarray, pd.Series]
        ) -> np.ndarray:

        """
        Ensures that a label array or Series is converted to one-hot encoding.

        `Flash Explanation:` *`Use this when you want to guarantee your labels are in one-hot format!`*  

        This function accepts either a NumPy array or a Pandas Series. If a Series is provided, 
        it first encodes categories as integer class codes, then converts them to one-hot.  
        If the input is already in one-hot format (2D with more than one column), it is returned unchanged.  
        Otherwise, raw integer labels are transformed into one-hot vectors.  

        Parameters:
            y: np.ndarray | pd.Series  
                The labels to be encoded. Can be:  
                - A Pandas Series of categorical values.  
                - A NumPy array of integer labels.  
                - A NumPy array already in one-hot encoded format.  

        Returns:
            np.ndarray  
                The labels in one-hot encoded format with shape (n_samples, n_classes).  

        Examples:
        --------
        >>> import numpy as np, pandas as pd
        >>> from tensorflow.keras.utils import to_categorical
        >>> from my_module import ensureOneHotEncoding

        #### Convert Pandas categorical Series to one-hot
        >>> y_series = pd.Series(["cat", "dog", "dog", "cat"])
        >>> encoded = ensureOneHotEncoding(y_series)
        >>> print(encoded)
        [[1. 0.]
         [0. 1.]
         [0. 1.]
         [1. 0.]]

        ---
        #### Convert integer NumPy array to one-hot
        >>> y_array = np.array([0, 1, 2])
        >>> encoded = ensureOneHotEncoding(y_array)
        >>> print(encoded)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]

        ---
        #### Pass-through when already one-hot
        >>> y_onehot = np.array([[1, 0], [0, 1], [1, 0]])
        >>> encoded = ensureOneHotEncoding(y_onehot)
        >>> print(np.allclose(encoded, y_onehot))
        True
        """

        if isinstance(y, pd.Series):
            arr = y.astype('category').cat.codes
            return to_categorical(arr)

        if len(y.shape) > 1 and y.shape[1] > 1:
            return y

        return to_categorical(y)
    
    @staticmethod
    def datasetPandasToNumpyNDArray(
            data: Union[pd.Series, pd.DataFrame], 
        ) -> np.ndarray:
        """
        Converts a Pandas Series or DataFrame into a NumPy array.

        `Flash Explanation:` *`Use this when you have a Pandas DataFrame or Series and need to transform it into a NumPy array!`*  

        This function leverages the built-in `.to_numpy()` method from Pandas, 
        ensuring efficient conversion of tabular or single-column data into 
        NumPy format.  

        Parameters:
            data : pd.Series | pd.DataFrame  
                The Pandas object containing the data to convert.  

        Returns:
            np.ndarray  
                A NumPy array containing the same data.  
        """
        return data.to_numpy()
            
    @staticmethod
    def convertNumpyNdArrayToGrayScale(images: np.ndarray) -> np.ndarray:
        """
        Converts a batch of RGB images into grayscale.  

        `Flash Explanation:` *`Use this when you want to convert RGB images into grayscale format!`*  

        This function checks if the last dimension is already grayscale 
        (single channel). If not, it computes the grayscale intensity 
        using the luminance-preserving formula with weights for red, 
        green, and blue channels.  

        Parameters:
            images : np.ndarray  
                A NumPy array of images with shape (n_samples, height, width, channels).  
                Channels can be 1 (grayscale) or 3 (RGB).  

        Returns:
            np.ndarray  
                Grayscale images with shape (n_samples, height, width, 1).  
        """
        
        if images.shape[-1] == 1:
            return images
        grayscale_images = np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])
        return np.expand_dims(grayscale_images, axis=-1)

    @staticmethod
    def resizeNpArray(array: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        """
        Resizes a NumPy image array to new dimensions.  

        `Flash Explanation:` *`Use this when you want to resize images (grayscale or RGB) to a new shape before feeding them into a model!`*  

        If the input already matches the target size, it is returned unchanged.  
        Handles both grayscale and RGB inputs, automatically expanding dimensions 
        when necessary for compatibility. Uses TensorFlow's `tf.image.resize` 
        for resizing.  

        Parameters:
            array : np.ndarray  
                Input array with shape (n_samples, height, width, channels) or 
                (n_samples, height, width).  
            new_height : int  
                Target height of the output images.  
            new_width : int  
                Target width of the output images.  

        Returns:
            np.ndarray  
                The resized array with shape (n_samples, new_height, new_width, channels).  
        """
        
        if array.shape[1:3] == (new_height, new_width):
            return array
        if len(array.shape) == 3:
            array = np.expand_dims(array, axis=-1)
        resized_array = tf.image.resize(array, (new_height, new_width)).numpy()
        
        return resized_array

    @staticmethod
    def convertNumpyNdArrayToRGB(images: np.ndarray) -> np.ndarray:
        """
        Converts grayscale images into RGB by repeating the channel.  

        `Flash Explanation:` *`Use this when you have grayscale images and need them in 
        RGB format for models or libraries that expect 3 channels!`*  

        If the input already has 3 channels, it is returned unchanged.  
        Otherwise, the grayscale channel is expanded and repeated across 
        the three RGB channels.  

        Parameters:
            images : np.ndarray  
                A NumPy array of images with shape (n_samples, height, width, channels).  
                Can be grayscale (1 channel) or RGB (3 channels).  

        Returns:
            np.ndarray  
                RGB images with shape (n_samples, height, width, 3).  
        """
        
        if images.shape[-1] == 3:
            return images
        
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
            
        return np.repeat(images, 3, axis=-1)

    @staticmethod
    def normalize_pixels(img_arr_data: np.ndarray) -> np.ndarray:
        """
        Normalize images loaded in a numpy array to [0, 1] dividing the pixel's values by 255.0.
        
        `Flash Explanation:` *`Use this when you want to keep the pixels between (0, 1)!`* 
        
        Parameters
            img_arr_data: np.ndarray
            Array of images with shape (n_samples, height, width, channels) or (n_samples, height, width).
        
        Returns
            np.ndarray
            Preprocessed array with shape (n_samples, height * width * channels).
        """

        if img_arr_data.ndim == 3:
            n_samples, height, width = img_arr_data.shape
            img_arr_data = img_arr_data.reshape((n_samples, height * width))
        elif img_arr_data.ndim == 4:
            n_samples, height, width, channels = img_arr_data.shape
            img_arr_data = img_arr_data.reshape((n_samples, height * width * channels))
        else:
            raise ValueError("Unsupported input array shape.")

        img_arr_data = img_arr_data.astype('float32') / 255.0
        
        return img_arr_data

    @staticmethod
    def minMaxScaler(x: pd.DataFrame | pd.Series | np.ndarray, min: float = 0, max: float = 1, return_scaler: bool = False) -> np.ndarray | tuple[MinMaxScaler, np.ndarray]:

        """
        Scale numerical features to a specified range using Min-Max normalization.

        `Flash Explanation:` *`Use this when you want to transform your data to a specific range (like [0, 1]) so that all features are comparable!`*

        Parameters
            x : pd.DataFrame | pd.Series | np.ndarray
                Input data to be scaled.
            min : float, default=0
                Desired minimum value of the transformed data.
            max : float, default=1
                Desired maximum value of the transformed data.
            return_scaler : bool, default=False
                If True, also returns the fitted MinMaxScaler object.

        Returns
            np.ndarray | tuple[MinMaxScaler, np.ndarray]
                Scaled data as a numpy array.  
                If `return_scaler=True`, returns a tuple containing `(scaler, scaled_data)`.
        """
        
        is_one_dimensional = isinstance(x, pd.Series) or (
            isinstance(x, np.ndarray) and x.ndim == 1
        )
        values = x.to_numpy().reshape(-1, 1) if isinstance(x, pd.Series) else x
        if is_one_dimensional:
            values = np.asarray(values).reshape(-1, 1)

        scaler = MinMaxScaler((min, max))
        scaler.fit(values)
        scaled_values = scaler.transform(values)
        if is_one_dimensional:
            scaled_values = scaled_values.ravel()

        if return_scaler:
            return scaler, scaled_values
        else:
            return scaled_values
    
    @staticmethod
    def minMaxScaleRevert(x: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
        
        """
        Revert data from Min-Max scaled values back to their original scale using a fitted MinMaxScaler.

        `Flash Explanation:` *`Use this when you want to bring your normalized data back to its original range! You must already have a fitted scaler`*

        Parameters
            x : np.ndarray
                Scaled data to be inverted.
            scaler : MinMaxScaler
                Fitted MinMaxScaler instance that was originally used for scaling.

        Returns
            np.ndarray
                Data transformed back to the original scale.
        """
        
        return scaler.inverse_transform(x)
    
    @staticmethod
    def labelEncoder(
            y: Union[np.ndarray, pd.Series, list], 
            return_encoder: bool = False
        ) -> np.ndarray | tuple[np.ndarray, LabelEncoder]:
        
        """
        Encode categorical labels into numerical format using scikit-learn's LabelEncoder.

        `Flash Explanation:` *`Use this when you need to convert class labels (like 'cat', 'dog') into numbers (like 0, 1) for model training!`*

        ## Parameters
            y : np.ndarray | pd.Series | list
                Array-like object with categorical labels.
            
            return_encoder : bool, default=False
                If True, also returns the fitted LabelEncoder object.

        ## Returns
            np.ndarray | tuple[np.ndarray, LabelEncoder]
                Encoded labels as a numpy array.  
                If `return_encoder=True`, returns a tuple containing `(encoded_labels, encoder)`.
        """
        
        le: LabelEncoder = LabelEncoder()
        le.fit(y)

        if return_encoder:
            return le.transform(y), le

        return le.transform(y)
    
    @staticmethod
    def labelDecoder(
            labels: Union[np.ndarray, pd.Series, list], 
            encoder: LabelEncoder
        ) -> np.ndarray:

        """
        Decode numerical labels back into their original categorical form using a fitted LabelEncoder.

        `Flash Explanation:` *`Use this when you want to transform your encoded labels (like 0, 1) back into the original categories (like 'cat', 'dog')! 
        You must already have a fitted encoder.`*

        Parameters
            labels : np.ndarray | pd.Series | list
                Encoded labels to be transformed back.
            encoder : LabelEncoder
                Fitted LabelEncoder instance used for the original encoding.

        Returns
            np.ndarray
                Decoded labels in their original categorical form.
        """

        return (encoder.inverse_transform(labels))