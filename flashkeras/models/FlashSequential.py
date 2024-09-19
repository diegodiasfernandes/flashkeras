from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
from flashkeras.models.FlashTransferLearning import FlashNet
from flashkeras.analysing.models import print_model_summary
from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

tasks_available = Literal['classification', 'regression']

class FlashSequential:
    def __init__(self, task: tasks_available) -> None:
        self.task: tasks_available = task

        self.model: Sequential = Sequential()
        self.layers = self.model.layers
        self.blocked: list[str] = []

        self.output_activation: Literal["sigmoid", "softmax"] = "sigmoid"
        self.output_loss: Literal["binary_crossentropy", "categorical_crossentropy"] = "binary_crossentropy"
        self.output_neurons: int = 1          

    def add(self, layer) -> None:       
        self._checkBlocked()

        self.model.add(layer)
        
        self.layers = self.model.layers

    def addTransferLearning(self, transferLayer: FlashNet) -> None: 
        '''
            Adds transfer learning layers created with FlashTransferLearning
        '''

        self._checkBlocked()

        if transferLayer.isFullNetwork: 
            self.model = transferLayer.network
            self.blocked.append('TransferLearning: Include Top was set. To maintain integrity of the flash, any modifycations to the architecture are blocked.')
        elif isinstance(transferLayer.network, Sequential):
            for l in transferLayer.network.layers:
                self.model.add(l)
        else:
            self.model.add(transferLayer.network)   

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

    def build(self, 
              data: Union[np.ndarray, pd.DataFrame, BatchIterator], 
              y: Union[np.ndarray, pd.Series, None] = None,
              add_auto_output_layer: bool = True
              ) -> None:
        ''' Automatically sets the `input_shape` and `output_layer` based on your data
        '''
        self._setOutputParams(y, data)

        if not self.model.inputs:
            self.setInputShape(preprocess.getInputShape(data))

        if add_auto_output_layer:
            self.model.add(Dense(self.output_neurons, self.output_activation))

    @overload
    def fit(self, 
            *, 
            x: np.ndarray | pd.DataFrame, 
            y: np.ndarray | pd.Series, 
            epochs: int = 10, 
            add_auto_output_layer: bool = False,
            steps_per_epoch: int | None = None,
            validation_data: tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]] | None = None
            ) -> None: ...
    @overload
    def fit(self, 
            *, 
            train_batches: BatchIterator, 
            epochs: int = 10, 
            add_auto_output_layer: bool = False,
            steps_per_epoch: int | None = None, 
            validation_data: BatchIterator | None = None
            ) -> None: ...

    def fit(self, 
            *, 
            x: Union[np.ndarray, pd.DataFrame, None] = None, 
            y: Union[np.ndarray, pd.Series, None] = None, 
            train_batches: Optional[BatchIterator] = None, 
            epochs: int = 10, 
            add_auto_output_layer: bool = False,
            validation_data: Optional[BatchIterator | tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]] = None, 
            steps_per_epoch: int | None = None 
            ) -> None:

        if train_batches is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot specify both `train_batches` and `x`/`y`.")
            data: Union[np.ndarray, pd.DataFrame, BatchIterator] = train_batches
        if train_batches is None:
            if x is None or y is None:
                raise ValueError("`x` and `y` must be provided unless using `train_batches`.")  
            if not ( (isinstance(x, pd.DataFrame) or isinstance(x, np.ndarray)) and (isinstance(y, pd.Series) or isinstance(y, np.ndarray)) ):
                raise ValueError("`x` must be of type (`pandas.DataFrame` or `np.ndarray`) and `y` (`pandas.Series` or `np.ndarray`).")  
            data = x

        self.build(data, y, add_auto_output_layer)

        if not self.model._is_compiled:
            self.compile()

        if train_batches is not None and (isinstance(train_batches, DirectoryIterator) or isinstance(train_batches, NumpyArrayIterator)):
            self.model.fit(train_batches, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
            return
        
        if x is not None and y is not None:
            '''y = preprocess.ensureOneHotEncoding(y)
            if isinstance(validation_data, tuple):
                new_y_test = preprocess.ensureOneHotEncoding(validation_data[1])
                validation_data = (validation_data[0], new_y_test)'''

            self.model.fit(x, y, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)

    def summary(self):
        try:
            self.model.summary()
        except:
            print("WARNING: The model is not built yet. Only the architecture will be shown.")
            print_model_summary(self.model)

    def loadModel(self, path_to_modelh5: str):
        print("This will Overwrite an existent model.")
        self.model = keras.models.load_model(path_to_modelh5)
        self.blocked.append('Loaded Model: You have loaded a full model. To maintain integrity of the flash, any modifications to the architecture are blocked.')

    def setInputShape(self, input_shape: tuple):
        new_model: Sequential = Sequential(InputLayer(input_shape=input_shape))
        for layer in self.model.layers: 
            new_model.add(layer)
        self.model = new_model
            
    def clearFlash(self) -> None:
        '''
            Reset every configuration made on flash model.
        '''
        self.model = Sequential()
        self.layers = self.model.layers
        self.blocked = []

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
        
    def _setOutputParams(self,
                y: Union[np.ndarray, pd.Series, None] = None, 
                image_batches: Optional[BatchIterator] = None, 
                ) -> Tuple[str, str, int]:
        
        sparse_or_not: str = ""
        if y is not None:
            if isinstance(y, pd.Series):
                num_classes = len(y.unique())
            elif isinstance(y, np.ndarray):
                if y.ndim == 1:
                    sparse_or_not += "sparse_"
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
            self.output_loss = sparse_or_not + "categorical_crossentropy"
            self.output_neurons = num_classes

        return self.output_activation, self.output_loss, self.output_neurons

    def _checkBlocked(self) -> None:
        if len(self.blocked) != 0:
            err_message = 'Possible errors:\n'
            for error in self.blocked:
                err_message = err_message + error + '\n'

            raise ValueError(err_message)
