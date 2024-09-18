from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
from flashkeras.models.transferlearning.FlashTransferLearning import FlashNet
from flashkeras.analysing.models import print_model_summary
from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

class FlashSequential:
    def __init__(self, task: Literal['classification', 'regression']) -> None:
        self.model: Sequential = Sequential()
        self.layers = self.model.layers
        self.blocked: list[str] = []

        self.output_activation: Literal["sigmoid", "softmax"] = "sigmoid"
        self.output_loss: Literal["binary_crossentropy", "categorical_crossentropy"] = "binary_crossentropy"
        self.output_neurons: int = 1          
            
    def clearFlash(self) -> None:
        '''
            Reset every configuration made on flash model.
        '''
        self.model = Sequential()
        self.layers = self.model.layers
        self.blocked = []

    def summary(self):
        try:
            self.model.summary()
        except:
            print("WARNING: The model is not built yet. Only the architecture will be shown.")
            print_model_summary(self.model)

    def getLayers(self) -> list[Sequential]:
        return self.model.layers

    def addTransferLearning(self, transferLayer: FlashNet) -> None: 
        '''
            Adds transfer learning layers created with FlashTransferLearning
        '''

        self.checkBlocked()

        if transferLayer.isFullNetwork: 
            self.model = transferLayer.network
            self.blocked.append('TransferLearning: Include Top was set. To maintain integrity of the flash, any modifycations to the architecture are blocked.')
        elif isinstance(transferLayer.network, Sequential):
            for l in transferLayer.network.layers:
                self.model.add(l)
        else:
            self.model.add(transferLayer.network)   

    def add(self, layer) -> None:       
        self.checkBlocked()

        self.model.add(layer)
        
        self.layers = self.model.layers

    def checkBlocked(self) -> None:
        if len(self.blocked) != 0:
            err_message = 'Possible errors:\n'
            for error in self.blocked:
                err_message = err_message + error + '\n'

            raise ValueError(err_message)
    
    def loadModel(self, path_to_modelh5: str):
        print("This will Overwrite an existent model.")
        self.model = keras.models.load_model(path_to_modelh5)
        self.blocked.append('Loaded Model: You have loaded a full model. To maintain integrity of the flash, any modifycations to the architecture are blocked.')

    def _optimizerMap(self, opt: str, lr: Optional[float]):
        if lr is None:
            lr = 0.0001
        if opt == "adam":
            return Adam(learning_rate=lr)
        elif opt == "nadam":
            return Nadam(learning_rate=lr)
        elif opt == "sgd":
            return SGD(learning_rate=lr)
        else:
            return Adam()
        
    def setOutputParams(self,
                y: Union[np.ndarray, pd.Series, None] = None, 
                image_batches: Optional[BatchIterator] = None, 
                ) -> Tuple[str, str, int]:
        
        if y is not None:
            if isinstance(y, pd.Series):
                num_classes = len(y.unique())
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    num_classes = len(np.unique(y))
                elif y.ndim == 2:
                    num_classes = y.shape[1]
        
        elif image_batches is not None:
            if isinstance(image_batches, DirectoryIterator):
                num_classes = image_batches.num_classes
            elif isinstance(image_batches, NumpyArrayIterator):
                num_classes = len(image_batches.y[0])
        else:
            raise ValueError("Either x or y must be provided.")
        
        # Determine activation and loss
        if num_classes == 2:
            self.output_activation = "sigmoid"
            self.output_loss = "binary_crossentropy"
            self.output_neurons = 1
        else:
            self.output_activation = "softmax"
            self.output_loss = "categorical_crossentropy"
            self.output_neurons = num_classes

        return self.output_activation, self.output_loss, self.output_neurons
    
    def setInputShape(self, input_shape: tuple):
        new_model: Sequential = Sequential(InputLayer(input_shape=input_shape))
        for layer in self.model.layers: 
            new_model.add(layer)
        self.model = new_model

    def compile(self,    
                optimizer: str | Any = "adam",
                learning_rate: float | None = None,
                loss: Any | None = None,
                metrics: Any | None = ['accuracy'],
                ) -> None:
        
        if isinstance(optimizer, str):
            opt = self._optimizerMap(optimizer, learning_rate)
        else:
            opt = optimizer

        if loss is None:
            self.model.compile(opt, self.output_loss, metrics)
        else:
            self.model.compile(opt, loss, metrics)

    @overload
    def fit(self, 
            *, 
            x: np.ndarray, 
            y: np.ndarray, 
            epochs: int = 10, 
            steps_per_epoch: int | None = None,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None
            ) -> None: ...
    @overload
    def fit(self, 
            *, 
            x: pd.DataFrame, 
            y: pd.Series, 
            epochs: int = 10, 
            steps_per_epoch: int | None = None, 
            validation_data: tuple[pd.DataFrame, pd.Series] | None = None
            ) -> None: ...
    @overload
    def fit(self, 
            *, 
            train_batches: BatchIterator, 
            epochs: int = 10, 
            steps_per_epoch: int | None = None, 
            validation_data: BatchIterator | None = None
            ) -> None: ...

    def fit(self, 
            *, 
            x: Union[np.ndarray, pd.DataFrame, None] = None, 
            y: Union[np.ndarray, pd.Series, None] = None, 
            train_batches: Optional[BatchIterator] = None, 
            epochs: int = 10, 
            validation_data: Optional[BatchIterator | tuple[Union[np.ndarray, pd.DataFrame] | Union[np.ndarray, pd.Series]]] = None, 
            steps_per_epoch: int | None = None 
            ) -> None:

        if train_batches is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot specify both `train_batches` and `x`/`y`.")
            data: Union[np.ndarray, pd.DataFrame, BatchIterator] = train_batches
        if train_batches is None:
            if x is None or y is None:
                raise ValueError("`x` and `y` must be provided unless using `train_batches`.")  
            if not ( (isinstance(x, pd.DataFrame) and isinstance(y, pd.Series)) or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) ):
                raise ValueError("`x` and `y` must be one of (`DataFrame` and `Series`) or (`ndarray` and `ndarray`).")  
            data = x

        self.setOutputParams(y, train_batches)

        if not self.model.inputs:
            self.setInputShape(preprocess.getInputShape(data))

        if not self.model._is_compiled:
            self.compile()

        self.model.add(Dense(self.output_neurons, self.output_activation))

        if train_batches is not None and (isinstance(train_batches, DirectoryIterator) or isinstance(train_batches, NumpyArrayIterator)):
            self.model.fit(train_batches, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
            return
        
        if x is not None and y is not None:
            x, y = preprocess.datasetToArray(x, y)

            self.model.fit(x, y, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)