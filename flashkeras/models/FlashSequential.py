from flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as preprocess
from flashkeras.models.FlashTransferLearning import FlashNet
from flashkeras.analysing.models import print_model_summary
from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

tasks_available = Literal['classification', 'regression']

class FlashSequential:
    """Build and manage a task-oriented Keras sequential model.

    `Flash Explanation:` *`Use this class to assemble, configure, train, evaluate, and reuse sequential Keras models with automatic task-specific output settings.`*

    Parameters:
        task: Model task, either ``"classification"`` or ``"regression"``.

    Attributes:
        model: Wrapped Keras ``Sequential`` model.
        layers: Current list of layers in the wrapped model.
        blocked: Messages describing why architecture modifications are blocked.
        output_activation: Activation selected for the inferred output layer.
        output_loss: Loss selected for the inferred task.
        output_neurons: Number of units selected for the output layer.
    """

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
        """Append a Keras layer to the model.

        `Flash Explanation:` *`Use this to add any compatible custom or keras-built-in layer before the model is blocked.`*
        
        Parameters:
            layer: Keras layer instance to append to the model.
        
        Returns:
            None. Updates the wrapped model architecture.
        
        Raises:
            ValueError: If architecture modifications are currently blocked.
        """
        self._checkBlocked()

        self.model.add(layer)
        
        self.layers = self.model.layers

    def addDense(
        self, 
        num_neurons: int, 
        activation: Literal[
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "linear",
            "elu",
            "selu",
            "gelu",
            "swish",
        ] = 'relu'
    ) -> None:       
        """Append a dense layer with the requested width and activation.

        `Flash Explanation:` *`Use this to add a fully connected layer without constructing ``Dense`` manually.`*
        
            Parameters:
                num_neurons: Number of units in the dense layer.
                activation: Activation function applied by the layer.
        
            Returns:
                None. Appends the dense layer to the wrapped model.
        
            Raises:
                ValueError: If architecture modifications are currently blocked.
        """
        self._checkBlocked()

        self.model.add(Dense(
            num_neurons,
            activation=activation
        ))
        
        self.layers = self.model.layers

    def addConvolution2D(
        self,
        filters: int,
        kernel_size: int | tuple[int, int] = (3, 3),
        activation: Literal[
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "linear",
            "elu",
            "selu",
            "gelu",
            "swish",
        ] = 'relu'
    ) -> None:
        """Append a two-dimensional convolutional layer.

        `Flash Explanation:` *`Use this to extract spatial features from image data before pooling, flattening, or dense layers.`*

        Parameters:
            filters: Number of convolution filters, also called output channels.
            kernel_size: Height and width of the convolution window. An integer
                creates a square window; a tuple can specify both dimensions.
            activation: Activation function applied by the layer.

        Returns:
            None. Appends a Keras ``Conv2D`` layer to the wrapped model.

        Raises:
            ValueError: If architecture modifications are currently blocked.
        """
        self._checkBlocked()

        self.model.add(Conv2D(
            filters,
            kernel_size,
            activation=activation
        ))

        self.layers = self.model.layers

    def addFlatten(self) -> None:       
        """Append a flattening layer to the model.

        `Flash Explanation:` *`Use this to convert spatial feature maps into a vector for dense layers.`*

        Parameters:
            None.
        
        Returns:
            None. Appends a Keras ``Flatten`` layer to the wrapped model.
        
        Raises:
            ValueError: If architecture modifications are currently blocked.
        """
        self._checkBlocked()

        self.model.add(Flatten())
        
        self.layers = self.model.layers

    def addTransferLearning(self, transferLayer: FlashNet) -> None: 
        '''
            Adds transfer learning layers created with FlashTransferLearning

            `Flash Explanation:` *`Use this to attach a pretrained feature extractor or full network.`*
            
            Parameters:
                transferLayer: Transfer-learning network created by ``FlashTransferLearning``.
            
            Returns:
                None. Adds the transfer-learning network to the wrapped model.
            
            Raises:
                ValueError: If architecture modifications are currently blocked.
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
        """Compile the wrapped Keras model with an optimizer, loss, and metrics.

        `Flash Explanation:` *`Use this to configure training; string optimizers are mapped to Keras optimizers.`*
        
        Parameters:
            optimizer: Optimizer name or configured optimizer instance.
            learning_rate: Optional learning rate used for named optimizers.
            loss: Loss function or loss name passed to Keras.
            metrics: Metrics passed to Keras; defaults depend on the task.
        
        Returns:
            None. Compiles the wrapped model and stores optimizer and metric settings.
        """

        if metrics is None:
            if self.task == 'classification':
                metrics = ['accuracy']
            elif self.task == 'regression':
                metrics = ['mae']

        if self.task == 'regression' and learning_rate is None:
            learning_rate = 0.001
        
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

          `Flash Explanation:` *`Use this to infer input and output configuration from representative data.`*
          
                    Parameters:
                            data: Representative input data or a Keras batch iterator.
                            y: Optional target labels used to infer output configuration.
                            auto_output_layer: Whether to append the inferred output layer.
          
                    Returns:
                            None. Configures the input shape and, optionally, the output layer.
        '''
        self._setOutputParams(y, data)

        if not self.model.inputs:
            self.setInputShape(preprocess.getInputShape(data))

        if auto_output_layer:

            if self.task == 'classification':
                self.model.add(Dense(self.output_neurons, self.output_activation))
            elif self.task == 'regression':
                self.model.add(Dense(self.output_neurons))

    def train(self, 
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
        """Build, compile, and train the wrapped model.

        `Flash Explanation:` *`Use this as the main training entry point for arrays, data frames, or iterators.`*
        
        Parameters:
            x: Training inputs or a Keras batch iterator.
            y: Training targets when ``x`` is array-like.
            epochs: Number of training epochs.
            auto_output_layer: Whether to add an inferred output layer before training.
            validation_data: Data used for validation during training.
            steps_per_epoch: Number of batches per training epoch.
            batch_size: Number of samples per batch for array-like inputs.
            verbose: Keras training verbosity setting.
            callbacks: Keras callbacks used during training.
            validation_split: Fraction of array-like data reserved for validation.
            shuffle: Whether to shuffle training data.
            class_weight: Optional class weights for classification.
            sample_weight: Optional per-sample weights.
            initial_epoch: Epoch from which to resume training.
        
        Returns:
            Any: Keras ``History`` object containing training results.
        """
        
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
        """Generate predictions for a batch of inputs.

        `Flash Explanation:` *`Use this to run the wrapped Keras model on arrays or iterators.`*
        
                Parameters:
                    x: Input samples, arrays, data frames, or a Keras iterator.
                    batch_size: Number of samples processed per prediction batch.
                    verbose: Keras prediction verbosity setting.
                    steps: Number of batches to process from an iterator.
        
                Returns:
                    Any: Predictions generated by the wrapped Keras model.
        """
        
        return self.model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps)

    def singlePredict(self, instance: Any):
        """Generate a prediction for one tabular or image instance.

        `Flash Explanation:` *`Use this to normalize one input's batch shape before prediction.`*
        
        Parameters:
            instance: One tabular or image sample to predict.
        
        Returns:
            np.ndarray: The predicted class or regression value for the sample.
        """
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
        """Print the model architecture summary.

        `Flash Explanation:` *`Use this to inspect the model even before it has been fully built.`*

        Parameters:
            None.

        Returns:
            None. Prints the model architecture summary.
        """
        try:
            self.model.summary()
        except:
            print("WARNING: The model is not built yet. Only the architecture will be shown.")
            print_model_summary(self.model)

    def loadModel(self, path_to_modelh5: str):
        """Load a saved Keras model and block unsafe architecture edits.

        `Flash Explanation:` *`Use this to restore a trained model from an H5-compatible path.`*
        
        Parameters:
            path_to_modelh5: Path to the saved Keras model file.
        
        Returns:
            None. Replaces the wrapped model and blocks architecture edits.
        """
        print("This will Overwrite an existent model.")
        self.model = keras.models.load_model(path_to_modelh5)
        self.blocked.append('Loaded Model: You have loaded a full model. To maintain integrity of the flash, any modifications to the architecture are blocked.')

    def setInputShape(self, input_shape: tuple):
        """Recreate the sequential model with an explicit input shape.

        `Flash Explanation:` *`Use this to establish the input layer before adding existing layers.`*
        
        Parameters:
            input_shape: Shape of one input sample, excluding the batch dimension.
        
        Returns:
            None. Rebuilds the wrapped sequential model with an input layer.
        """
        new_model: Sequential = Sequential(InputLayer(input_shape=input_shape))
        for layer in self.model.layers: 
            new_model.add(layer)
        self.model = new_model
            
    def clearFlash(self) -> None:
        '''
            Reset every configuration made on flash model.

            `Flash Explanation:` *`Use this to discard the current architecture and unblock the wrapper.`*

            Parameters:
                None.
            
            Returns:
                None. Resets the model architecture and blocked state.
        '''
        self.model = Sequential()
        self.layers = self.model.layers
        self.blocked = []

    def _optimizerMap(self, opt: str, lr: Optional[float]):
        """Map a supported optimizer name and learning rate to a Keras optimizer.

        `Flash Explanation:` *`Use this internal helper to turn user-friendly optimizer options into instances.`*
        
        Parameters:
            opt: Optimizer name, such as ``"adam"``, ``"nadam"``, or ``"sgd"``.
            lr: Optional learning rate; defaults to ``0.0001`` when omitted.
        
        Returns:
            Any: Configured Keras optimizer instance.
        """
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
        """Return whether every nested label array contains one value.

        `Flash Explanation:` *`Use this internal helper to distinguish sparse labels from one-hot labels.`*
        
        Parameters:
            arr: Array or Series containing nested label values.
        
        Returns:
            bool: ``True`` when every nested value contains one item.
        """
        for subarray in arr:
            if subarray.size != 1:
                return False
        return True

    def _setOutputParams(self,
                y: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None, 
                image_batches: Optional[BatchIterator] = None, 
                ) -> None:
        """Infer output activation, loss, and neuron count from the task labels.

        `Flash Explanation:` *`Use this internal helper before building a task-compatible output layer.`*
        
        Parameters:
            y: Optional tabular labels used to infer output settings.
            image_batches: Optional Keras iterator containing labels.
        
        Returns:
            None. Updates output activation, loss, and neuron count attributes.
        
        Raises:
            ValueError: If required labels or input metadata are unavailable.
        """
        
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
                    if getattr(image_batches, "class_mode", "categorical") == "sparse":
                        sparse_or_not += "sparse_"

                elif isinstance(image_batches, NumpyArrayIterator):
                    iterator_y = image_batches.y
                    if iterator_y.ndim == 1 or self._array_of_unit_arrays(iterator_y):
                        sparse_or_not += "sparse_"
                        num_classes = len(np.unique(iterator_y))
                    elif iterator_y.ndim == 2:
                        num_classes = iterator_y.shape[1]
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
        """Raise an error when model mutation has been blocked by a full model load.

        `Flash Explanation:` *`Use this internal guard to preserve the integrity of protected architectures.`*

        Parameters:
            None.
        
        Returns:
            None when architecture edits are allowed.
        
        Raises:
            ValueError: If the model is protected from architecture changes.
        """
        if len(self.blocked) != 0:
            err_message = 'Possible errors:\n'
            for error in self.blocked:
                err_message = err_message + error + '\n'

            raise ValueError(err_message)
