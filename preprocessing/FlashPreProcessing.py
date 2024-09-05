import cv2
from keras.models import Sequential # type: ignore
from keras.layers import InputLayer # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.preprocessing.image import DirectoryIterator # type: ignore
import keras # type: ignore
from typing import overload, Union, Tuple, Literal
import numpy as np
import pandas as pd

class FlashPreProcessing:

    @staticmethod
    def ensureOneHotEncoding(
                y: Union[np.ndarray, pd.Series, None] = None
                ) -> Union[np.ndarray, pd.Series, None]:
        
        new_y = y
        
        if new_y is not None:
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
            if new_y is not None:
                if isinstance(new_y, pd.Series) or isinstance(new_y, np.ndarray):
                    if new_y.ndim == 2 and new_y.shape[1] == 1:
                        new_y = new_y.ravel()  # Flatten y if it's a 2D array with one column
                    if np.unique(new_y).tolist() != [0, 1]:
                        new_y = (new_y == np.max(new_y)).astype(int)  # Ensure y is binary (0 and 1)
                        
        else:
            if new_y is not None:
                if isinstance(new_y, pd.Series) or isinstance(new_y, np.ndarray):
                    if new_y.ndim == 1 or new_y.shape[1] == 1:  # If y is not one-hot encoded
                        new_y = np.eye(num_classes)[new_y.astype(int)]

        return new_y
    
    @staticmethod
    def adjustXY(
            x: Union[np.ndarray, pd.DataFrame, None] = None, 
            y: Union[np.ndarray, pd.Series, None] = None
            ) -> tuple[Union[np.ndarray, pd.DataFrame, None], Union[np.ndarray, pd.Series, None]]:
        
        new_y = FlashPreProcessing.ensureOneHotEncoding(y)
        new_x = x

        if isinstance(new_x, pd.DataFrame):
            new_x = new_x.values  # Convert DataFrame to NumPy array

        if isinstance(new_y, pd.Series):
            new_y = new_y.to_numpy()  # Convert Series to NumPy array
        
        return new_x, new_y
    
    @staticmethod
    def reshapeNumpyImages(x: np.ndarray, 
                           input_shape: tuple[int, int, int] = (224, 224, 3)
                           ) -> np.ndarray:
        new_x = x
        new_x = np.expand_dims(new_x, axis=-1)  # Transforma (60000, 28, 28) em (60000, 28, 28, 1)

        new_x_resized = np.zeros((new_x.shape[0], input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)

        for i in range(new_x.shape[0]):
            resized_img = cv2.resize(new_x[i], (input_shape[1], input_shape[0]))  # Redimensiona para (224, 224)
            new_x_resized[i] = np.repeat(resized_img, input_shape[2], axis=-1)  # Converte para (224, 224, 3)
        
        return new_x_resized