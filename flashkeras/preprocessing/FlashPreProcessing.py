from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

class FlashPreProcessing:

    @staticmethod
    def train_test_split(*arrays: Any, 
                         test_split: float | None = None,
                         random_state: int | None = None,
                         ) -> list:
        """Split arrays or matrices into random train and test subsets.

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
    def stackDataFrames(matrix_a: pd.DataFrame | np.ndarray, matrix_b: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(matrix_a, pd.DataFrame) and isinstance(matrix_b, pd.DataFrame):
            return pd.concat([matrix_a, matrix_b], ignore_index=True)
        
        elif isinstance(matrix_a, np.ndarray) and isinstance(matrix_b, np.ndarray):
            return np.vstack((matrix_a, matrix_b))
        
        elif isinstance(matrix_a, pd.DataFrame) and isinstance(matrix_b, np.ndarray):
            return np.vstack((matrix_a.values, matrix_b))
        
        elif isinstance(matrix_a, np.ndarray) and isinstance(matrix_b, pd.DataFrame):
            return np.vstack((matrix_a, matrix_b.values))
        
        else:
            raise ValueError("matrix_a and matrix_b should be of type `pandas.DataFrame` or `numpy.ndarray`")

    @staticmethod
    def getInputShape(data: Union[np.ndarray, pd.DataFrame, DirectoryIterator, NumpyArrayIterator]) -> tuple:
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

    @staticmethod
    def ensureOneHotEncoding(
                y: Union[np.ndarray, pd.Series]
                ) -> np.ndarray:

        if isinstance(y, pd.Series):
            arr = y.astype('category').cat.codes
            return to_categorical(arr)

        if len(y.shape) > 1 and y.shape[1] > 1:
            return y

        return to_categorical(y)
    
    @staticmethod
    def datasetToArray(
            x: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]
            ) -> tuple[np.ndarray, np.ndarray]:
        
        new_y = FlashPreProcessing.ensureOneHotEncoding(y)
        new_x = x

        if isinstance(new_x, pd.DataFrame):
            new_x = new_x.values  # Convert DataFrame to NumPy array

        if isinstance(new_y, pd.Series):
            new_y = new_y.to_numpy()  # Convert Series to NumPy array
        
        return new_x, new_y
    
    @staticmethod
    def convertNdArrayToGrayScale(images: np.ndarray) -> np.ndarray:
        if images.shape[-1] == 1:
            return images
        grayscale_images = np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])
        return np.expand_dims(grayscale_images, axis=-1)

    @staticmethod
    def resizeNpArray(array: np.ndarray, new_size1: int, new_size2: int) -> np.ndarray:
        if array.shape[1:3] == (new_size1, new_size2):
            return array
        if len(array.shape) == 3:
            array = np.expand_dims(array, axis=-1)
        resized_array = tf.image.resize(array, (new_size1, new_size2)).numpy()
        
        return resized_array

    @staticmethod
    def convertNdArrayToRGB(images: np.ndarray) -> np.ndarray:
        """
            Convert a batch of grayscale images (shape: (num_images, height, width) or 
            (num_images, height, width, 1)) into RGB format (shape: (num_images, height, width, 3)).
            
            Parameters:
            images: np.ndarray, Input image batch of shape (num_images, height, width) or 
                    (num_images, height, width, 1)
            
            Returns:
            np.ndarray, A batch of images with shape (num_images, height, width, 3).
        """

        if images.shape[-1] == 3:
            return images
        
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
            
        return np.repeat(images, 3, axis=-1)
    
    @staticmethod
    def minMaxScaler(x: pd.DataFrame | np.ndarray, min: float = 0, max: float = 1) -> np.ndarray:
        scaler = MinMaxScaler((min, max))
        return scaler.fit_transform(x)