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
    def stackDataFrames(matrix_a: pd.DataFrame | pd.Series | np.ndarray, 
                        matrix_b: pd.DataFrame | pd.Series | np.ndarray
                        ) -> pd.DataFrame | pd.Series | np.ndarray:
        
        def to_2d_array(data):
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
    def stackImageDatasets(data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray: ...
    
    @overload
    @staticmethod
    def stackImageDatasets(data_a: Tuple[np.ndarray, np.ndarray], data_b: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: ...
    
    @staticmethod
    def stackImageDatasets(data_a: np.ndarray | Tuple[np.ndarray, np.ndarray], 
                       data_b: np.ndarray | Tuple[np.ndarray, np.ndarray]
                       ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        
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
    
    @staticmethod
    def labelEncoder(y: Union[np.ndarray, pd.Series, list], 
                     return_encoder: bool = False
                     ) -> np.ndarray | tuple[np.ndarray, LabelEncoder]:
        
        le: LabelEncoder = LabelEncoder()
        le.fit(y)

        if return_encoder:
            return le.transform(y), le

        return le.transform(y)
    
    @staticmethod
    def labelDecoder(labels: Union[np.ndarray, pd.Series, list], 
                     encoder: LabelEncoder
                     ) -> np.ndarray:

        return (encoder.inverse_transform(labels))