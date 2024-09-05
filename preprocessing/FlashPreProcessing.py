from utils.otherimports import *
from utils.kerasimports import *
from utils.typehints import *

class FlashPreProcessing:

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
                ):
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            return pd.get_dummies(y).values
        
        return pd.get_dummies(y).values
    
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
        return tf.image.resize(array, (new_size1, new_size2)).numpy()

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

        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1) 
        
        rgb_images = np.repeat(images, 3, axis=-1)
        
        return rgb_images