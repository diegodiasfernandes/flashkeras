from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

class FlashPreProcessing:

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
    def resizeNpArray(array: np.ndarray, new_size1: int, new_size2: int) -> np.ndarray:
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