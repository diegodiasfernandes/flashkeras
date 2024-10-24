from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
from flashkeras.models.FlashTransferLearning import FlashNet
from flashkeras.analysing.models import print_model_summary
from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

tasks_available = Literal['classification', 'regression']

class FlashSequential:
    def __init__(self, task: Literal['classification', 'regression']) -> None:
        self.task: Literal['classification', 'regression'] = task

        self.model: Sequential = Sequential()
        self.layers = self.model.layers
        self.blocked: list[str] = []

        self.output_activation: Literal["sigmoid", "softmax"] = "sigmoid"
        self.output_loss: str = "binary_crossentropy"
        self.output_neurons: int = 1   

        self.optimizer: Any = None
        self.metrics: Any = None       

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
                metrics: Any = None,
                ) -> None:

        if metrics is None:
            if self.task == 'classification':
                metrics = ['accuracy']
            elif self.task == 'regression':
                metrics = ['mae']
        
        if isinstance(optimizer, str):
            opt = self._optimizerMap(optimizer, learning_rate)
        else:
            opt = optimizer

        self.model.compile(opt, loss, metrics)

        self.optimizer = opt
        self.metrics = metrics

    def build(self, 
              data: Union[np.ndarray, pd.DataFrame, BatchIterator], 
              y: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None,
              auto_output_layer: bool = True
              ) -> None:
        ''' Automatically sets the `input_shape` and `output_layer` based on your data
        '''
        self._setOutputParams(y, data)

        if not self.model.inputs:
            self.setInputShape(preprocess.getInputShape(data))

        if auto_output_layer:

            if self.task == 'classification':
                self.model.add(Dense(self.output_neurons, self.output_activation))
            elif self.task == 'regression':
                self.model.add(Dense(self.output_neurons))

    def fit(self, 
            x: Any | None = None,
            y: Any | None = None,
            epochs: int = 1,
            auto_output_layer: bool = False,
            validation_data: Any | None = None,
            steps_per_epoch: Any | None = None,
            batch_size: Any | None = None,
            verbose: str = "auto",
            callbacks: Any | None = None,
            validation_split: float = 0,
            shuffle: bool = True,
            class_weight: Any | None = None,
            sample_weight: Any | None = None,
            initial_epoch: int = 0
            ) -> Any:
        
        self.build(x, y, auto_output_layer)

        if not self.model._is_compiled:
            self.compile(loss=self.output_loss)
        
        if not self.model.loss:
            self.compile(optimizer=self.model.optimizer,
                         loss=self.output_loss,
                         metrics=self.metrics)

        if isinstance(x, (DirectoryIterator, NumpyArrayIterator)):
            history = self.model.fit(x, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_split=validation_split,
                           shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch)
            return history
        
        if x is not None and y is not None:
            history = self.model.fit(x, y, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_split=validation_split,
                           shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch)
            return history
        elif x is not None and y is None:
            history = self.model.fit(x, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch,
                           batch_size=batch_size, verbose=verbose, callbacks=callbacks, validation_split=validation_split,
                           shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch)
            return history
        
    def predict(self,
                x: Any,
                batch_size: Any | None = None,
                verbose: str = "auto",
                steps: Any | None = None,
                # callbacks: Any | None = None,
                # max_queue_size: int = 10,
                # workers: int = 1,
                # use_multiprocessing: bool = False
                ) -> Any:
        
        return self.model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps)

    def singlePredict(self, instance: Any):
        if isinstance(instance, np.ndarray):
            if len(instance.shape) == 2:
                instance = np.expand_dims(instance, axis=-1)

            if len(instance.shape) == 3:
                instance = np.expand_dims(instance, axis=0)
            elif len(instance.shape) == 1:
                instance = np.expand_dims(instance, axis=0)

        elif isinstance(instance, pd.DataFrame):
            if len(instance.shape) == 1 or instance.shape[0] == 1:
                instance = instance.to_numpy().reshape(1, -1)

        prediction = cast(np.ndarray, self.predict(instance))

        if len(prediction.shape) == 2:
            return np.argmax(prediction, axis=1)
        
        elif len(prediction.shape) == 1:
            return prediction

        return prediction

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
        
    def _array_of_unit_arrays(self, arr: np.ndarray | pd.Series) -> bool:
        for subarray in arr:
            if subarray.size != 1:
                return False
        return True

    def _setOutputParams(self,
                y: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None, 
                image_batches: Optional[BatchIterator] = None, 
                ) -> None:
        
        if self.task == 'classification':
            sparse_or_not: str = ""
            if y is not None:
                if isinstance(y, (pd.Series, np.ndarray)):
                    if y.ndim == 1 or self._array_of_unit_arrays(y):
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
        
        elif self.task == 'regression':
            if y is None:
                raise ValueError('``y`` cannot be None value.')
            
            if isinstance(y, (pd.Series, np.ndarray)):
                if y.ndim > 1:
                    self.output_neurons = y.shape[1]
                else:
                    self.output_neurons = 1
            else:
                self.output_neurons = len(y.columns)
            
            self.output_loss = 'mse'

    def _checkBlocked(self) -> None:
        if len(self.blocked) != 0:
            err_message = 'Possible errors:\n'
            for error in self.blocked:
                err_message = err_message + error + '\n'

            raise ValueError(err_message)
