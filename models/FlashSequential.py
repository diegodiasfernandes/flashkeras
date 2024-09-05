from keras.models import Sequential # type: ignore
from keras.layers import InputLayer, Dense # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.preprocessing.image import DirectoryIterator # type: ignore
from preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
import keras # type: ignore
from typing import overload, Union, Tuple, Any
import numpy as np
import pandas as pd # type: ignore

class FlashSequential:
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        self.model: Sequential = self._initializeInput(input_shape)
        self.layers = self.model.layers
        self.blocked: list[str] = []

    def _initializeInput(self, input_shape) -> Sequential:
        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        return model
    
    @overload
    def getInputShape(self, data: np.ndarray | None = None) -> tuple: ...
    @overload
    def getInputShape(self, data: DirectoryIterator | None = None) -> tuple: ...

    def getInputShape(self,
                      data: Union[np.ndarray, pd.DataFrame, DirectoryIterator, None] = None
                      ) -> tuple:
            
            if data is None:
                raise ValueError("parameter should be of types: (pd.DataFrame or np.ndarray or DirectoryIterator)")
            
            input_shape: tuple[int, int, int] | tuple[int] = (1, 1, 1)

            if isinstance(data, DirectoryIterator):
                input_shape = (data.target_size[0], data.target_size[1], 3)

            elif (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)) and data.ndim < 3:
                temp_data = data
                temp_data = pd.DataFrame(temp_data)
                input_shape = (temp_data.shape[1], )
            elif isinstance(data, np.ndarray) and data.ndim >= 3:
                shape = data[0].shape
                if len(shape) == 2: input_shape = (shape[0], shape[1], 1)
                else: input_shape = (shape[0], shape[1], 3)                
            
            return input_shape

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
                          train_batches: Union[DirectoryIterator, None] = None
                          ) -> None:
        
        output_activation, output_loss, output_neurons = self.getOutputParams(y, train_batches)
        opt = self._optimizerMap(opt[0], opt[1])
                
        self.model.add(Dense(output_neurons, output_activation))
        self.model.compile(opt, output_loss, metrics)

    @overload
    def fit(self, 
            *, 
            x: np.ndarray, 
            y: np.ndarray, 
            epochs: int = 10, 
            optimizer: str = "adam", 
            learning_rate: float = 0.001, 
            metrics: list = ['accuracy'], 
            steps_per_epoch: int | None = None,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None
            ) -> None: ...

    @overload
    def fit(self, 
            *, 
            x: pd.DataFrame, 
            y: pd.Series, 
            epochs: int = 10, 
            optimizer: str = "adam", 
            learning_rate: float = 0.001, 
            metrics: list = ['accuracy'], 
            steps_per_epoch: int | None = None, 
            validation_data: tuple[pd.DataFrame, pd.Series] | None = None
            ) -> None: ...

    @overload
    def fit(self, 
            *, 
            train_batches: DirectoryIterator, 
            epochs: int = 10, 
            optimizer: str = "adam", 
            learning_rate: float = 0.001, 
            metrics: list = ['accuracy'], 
            steps_per_epoch: int | None = None, 
            validation_data: DirectoryIterator | None = None
            ) -> None: ...

    def fit(self, 
            *, 
            x: Union[np.ndarray, pd.DataFrame, None] = None, 
            y: Union[np.ndarray, pd.Series, None] = None, 
            train_batches: Union[DirectoryIterator, None] = None, 
            epochs: int = 10, 
            validation_data: Union[DirectoryIterator, tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]], None] = None, 
            optimizer: str = "adam", 
            learning_rate: float = 0.001, 
            metrics: list = ['accuracy'], 
            steps_per_epoch: int | None = None 
            ) -> None:

        if train_batches is not None and (x is not None or y is not None):
            raise ValueError("Cannot specify both `train_batches` and `x`/`y`.")
        
        self._setOutputConfigs(
            opt = (optimizer, learning_rate), 
            metrics = metrics, 
            x = x, 
            y = y, 
            train_batches=train_batches
        )

        if train_batches is not None and isinstance(train_batches, DirectoryIterator):
            self.model.fit(train_batches=train_batches, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
            return
        
        if x is None or y is None:
            raise ValueError("`x` and `y` must be provided unless using `train_batches`.")
        
        if not ( (isinstance(x, pd.DataFrame) and isinstance(y, pd.Series)) or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) ):
            raise ValueError("`x` and `y` must be one of (`DataFrame` and `Series`) or (`ndarray` and `ndarray`).")
        
        x, y = preprocess.adjustXY(x, y)

        self.model.fit(x=x, y=y, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)

