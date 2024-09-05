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
                ) -> Union[np.ndarray, pd.Series]:
        
        new_y = y
        
        if isinstance(new_y, pd.Series):
            num_classes = len(new_y.unique())
        elif isinstance(new_y, np.ndarray):
            if new_y.ndim == 1:  # Case where y is a 1D array of labels
                num_classes = len(np.unique(new_y))
            elif new_y.ndim == 2:  # Case where y is one-hot encoded
                num_classes = new_y.shape[1]
        else:
            raise ValueError("Unsupported y type.")
                
        if num_classes == 2:
            if isinstance(new_y, pd.Series) or isinstance(new_y, np.ndarray):
                if new_y.ndim == 2 and new_y.shape[1] == 1:
                    new_y = new_y.ravel()  # Flatten y if it's a 2D array with one column
                if np.unique(new_y).tolist() != [0, 1]:
                    new_y = cast(np.ndarray, (new_y == np.max(new_y)).astype(int))  # Ensure y is binary (0 and 1)
                        
        else:
            if isinstance(new_y, pd.Series) or isinstance(new_y, np.ndarray):
                if new_y.ndim == 1 or new_y.shape[1] == 1:  # If y is not one-hot encoded
                    new_y = np.eye(num_classes)[new_y.astype(int)]

        return new_y
    
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