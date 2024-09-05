from preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
from models.transferlearning.FlashTransferLearning import FlashNet
from analysing.models import print_model_summary
from utils.otherimports import *
from utils.kerasimports import *
from utils.typehints import *

class FlashSequential:
    def __init__(self) -> None:
        self.model: Sequential = Sequential()
        self.layers = self.model.layers
        self.blocked: list[str] = []
        self.builded = False             
            
    def clearFlash(self) -> None:
        '''
            Reset every configuration made on flash model.
        '''
        self.model = Sequential()
        self.layers = self.model.layers
        self.blocked = []
        self.builded = False

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

    def _optimizerMap(self, opt: str, lr: float):
        if opt == "adam":
            return Adam(learning_rate=lr)
        elif opt == "nadam":
            return Nadam(learning_rate=lr)
        elif opt == "sgd":
            return SGD(learning_rate=lr)
        
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
            raise ValueError("Either x or y must be provided.")
        
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
                          train_batches: Union[DirectoryIterator, None] = None, 
                          y: Union[np.ndarray, pd.Series, None] = None
                          ) -> None:
        
        output_activation, output_loss, output_neurons = self.getOutputParams(y, train_batches)
        opt = self._optimizerMap(opt[0], opt[1])
                
        self.model.add(Dense(output_neurons, output_activation))
        self.model.compile(opt, output_loss, metrics)
        self.builded = True

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

        if train_batches is not None:
            if x is not None or y is not None:
                raise ValueError("Cannot specify both `train_batches` and `x`/`y`.")
            data: Union[np.ndarray, pd.DataFrame, DirectoryIterator] = train_batches
        if train_batches is None:
            if x is None or y is None:
                raise ValueError("`x` and `y` must be provided unless using `train_batches`.")  
            if not ( (isinstance(x, pd.DataFrame) and isinstance(y, pd.Series)) or (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) ):
                raise ValueError("`x` and `y` must be one of (`DataFrame` and `Series`) or (`ndarray` and `ndarray`).")  
            data = x

        if not self.builded:
            new_model: Sequential = Sequential(InputLayer(input_shape=preprocess.getInputShape(data)))
            for layer in self.model.layers: 
                new_model.add(layer)
            self.model = new_model

        self._setOutputConfigs(
            opt = (optimizer, learning_rate), 
            metrics = metrics, 
            y = y, 
            train_batches=train_batches
        )

        if train_batches is not None and isinstance(train_batches, DirectoryIterator):
            self.model.fit(train_batches, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)
            return
        
        if x is not None and y is not None:
            x, y = preprocess.datasetToArray(x, y)

            self.model.fit(x, y, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)