from keras.models import Sequential # type: ignore
from keras.layers import InputLayer # type: ignore
import keras # type: ignore

class FlashSequential:
    def __init__(self,         
                 input_shape_or_num_features: tuple[int, int, int] | int
                 ) -> None:
        
        self.input_shape = input_shape_or_num_features
        self.model: Sequential = self._initializeInput()
        self.blocked: list[str] = []

    def _initializeInput(self) -> Sequential:
        model = Sequential()

        if type(self.input_shape) != int:
            model.add(InputLayer(input_shape=self.input_shape))
        else:
            model.add(InputLayer(input_shape=(self.input_shape, )))

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
                    self.blocked.append('TransferLearning: Include Top was set. To maintain integrity of the flash, any modifycations to the architecture are blocked.\n')
                elif type(network) == type(Sequential):
                    for l in network.layers:
                        self.model.add(l)
                else:
                    self.model.add(network)   
                return True
            return False         

        if len(self.blocked) != 0:
            err_message = ''
            for error in self.blocked:
                err_message += error

            raise ValueError(err_message)

        if not transferedLearn():
            self.model.add(layer)
    
    def loadModel(self, path_to_modelh5: str):
        print("This will Overwrite an existent model.")
        self.model = keras.models.load_model(path_to_modelh5)

