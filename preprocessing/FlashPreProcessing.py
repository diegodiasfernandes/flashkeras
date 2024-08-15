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
                ) -> Tuple[str, str, int, Union[np.ndarray, pd.Series, None]]:
        
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
            new_y = new_y.values  # Convert Series to NumPy array
        
        return new_x, new_y