from keras.models import Sequential # type: ignore
from keras.layers import InputLayer, Dense # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.preprocessing.image import DirectoryIterator # type: ignore
from preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
import keras # type: ignore
from typing import overload, Union, Tuple
import numpy as np
import pandas as pd

class FlashSequential:
    def __init__(self) -> None:
        self.input_shape: tuple = (224, 224, 3)
        self.model: Sequential = self._initializeInput()
        self.layers = self.model.layers
        self.blocked: list[str] = []

    def _initializeInput(self) -> Sequential:
        model = Sequential()

        model.add(InputLayer(input_shape=self.input_shape))

        return model

    def clearFlash(self):
        self.model = self._initializeInput()

    def summary(self):
        self.model.summary()
    
    def getLayers(self) -> list[Sequential]:
        return self.model.layers

    def add(self, layer) -> None:
        def transferedLearn() -> bool:
            if type(layer) == tuple:
                network, is_full_model = layer
                if is_full_model: 
                    self.model = network
                    self.blocked.append('TransferLearning: Include Top was set. To maintain integrity of the flash, any modifycations to the architecture are blocked.')
                elif type(network) == type(Sequential):
                    for l in network.layers:
                        self.model.add(l)
                else:
                    self.model.add(network)   
                return True
            return False         

        if len(self.blocked) != 0:
            err_message = 'Possible errors:\n'
            for error in self.blocked:
                err_message = err_message + error + '\n'

            raise ValueError(err_message)

        if not transferedLearn():
            self.model.add(layer)
        
        self.layers = self.model.layers
    
    def loadModel(self, path_to_modelh5: str):
        print("This will Overwrite an existent model.")
        self.model = keras.models.load_model(path_to_modelh5)
        self.blocked.append('Loaded Model: You have loaded a full model. To maintain integrity of the flash, any modifycations to the architecture are blocked.')

    def _optimizerMap(self, opt: str, lr: float):
        if opt == "adam":
            return Adam(learning_rate=lr)
        
    def getOutputParams(self,
                y: Union[np.ndarray, pd.Series, None] = None, 
                image_batches: Union[DirectoryIterator, None] = None, 
                ) -> Tuple[str, str, int]:
        
        # Determine the number of classes (output_neurons)
        if y is not None:
            if isinstance(y, pd.Series):
                num_classes = len(y.unique())
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:  # Case where y is a 1D array of labels
                    num_classes = len(np.unique(y))
                elif y.ndim == 2:  # Case where y is one-hot encoded
                    num_classes = y.shape[1]
            else:
                raise ValueError("Unsupported y type.")
        
        elif image_batches is not None:
            num_classes = image_batches.num_classes
        else:
            raise ValueError("Either x and y or image_batches must be provided.")
        
        # Determine activation and loss
        if num_classes == 2:
            activation = "sigmoid"
            loss = "binary_crossentropy"
            output_neurons = 1
        else:
            activation = "softmax"
            loss = "categorical_crossentropy"
            output_neurons = num_classes

        return activation, loss, output_neurons
    
    def _setOutputConfigs(self,
                          opt: tuple[str, float],
                          metrics: list,
                          x: Union[np.ndarray, pd.DataFrame, None] = None, 
                          y: Union[np.ndarray, pd.Series, None] = None, 
                          image_batches: Union[DirectoryIterator, None] = None
                          ) -> None:
        
        output_activation, output_loss, output_neurons = self.getOutputParams(y, image_batches)
        opt = self._optimizerMap(opt[0], opt[1])
        
        if image_batches is not None:
            self.model.input_shape = image_batches.image_shape
        elif isinstance(x, np.ndarray):
            temp_x = x
            temp_x = pd.DataFrame(temp_x)
            self.model.input_shape = (temp_x.shape[1], )
            
        self.model.add(Dense(output_neurons, output_activation))
        self.model.compile(opt, output_loss, metrics)

    @overload
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 32, epochs: int = 10) -> None: ...
    @overload
    def fit(self, x: pd.DataFrame, y: pd.Series, batch_size: int = 32, epochs: int = 10) -> None: ...
    @overload
    def fit(self, image_batches: DirectoryIterator, epochs: int = 10) -> None: ...

    def fit(self,
            x: Union[np.ndarray, pd.DataFrame, None] = None, 
            y: Union[np.ndarray, pd.Series, None] = None, 
            image_batches: Union[DirectoryIterator, None] = None, 
            epochs: int = 10,
            validation_data: tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]] | None = None,
            optimizer: str = "adam",
            learning_rate: float = 0.001,
            metrics: list = ['accuracy'],
            steps_per_epoch: int | None = None 
            ) -> None:
        
        self._setOutputConfigs(
            opt = (optimizer, learning_rate), 
            metrics = metrics, 
            x = x, 
            y = y, 
            image_batches=image_batches
        )

        if image_batches is not None:
            self.model.fit(image_batches, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
            return
        
        x, y = preprocess.adjustXY(x, y)

        self.model.fit(x, y, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)

