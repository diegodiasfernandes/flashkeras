from keras.applications import MobileNet, ResNet50, Xception, VGG16 # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Input # type: ignore
import numpy as np # type: ignore
import keras # type: ignore

class TransferLearning:
    def __init__(self,         
                 input_shape: tuple[int, int, int],
                 include_top: bool = True,
                 weights: str = "imagenet",
                 freeze: bool | int = False,
                 use_only_few_layers: bool | int = False
                 ) -> None:
        
        self.input_shape: tuple[int, int, int] = input_shape
        self.include_top: bool = include_top
        self.weights: str = weights
        self.freeze: bool | int = freeze
        self.use_only_few_layers: bool | int = use_only_few_layers
    
    def freezeLayers(self, model: Model) -> Model:
        if not self.freeze:
            return model

        inputs = model.input
        x = inputs

        if type(self.freeze) == bool and self.freeze == True:
            for layer in model.layers:
                layer.trainable = False
                x = layer(x)
            return Model(inputs=inputs, outputs=x)

        count = 0
        for layer in model.layers:
            if count < self.freeze:  # freezing
                layer.trainable = False
            x = layer(x)
            count += 1

        return Model(inputs=inputs, outputs=x)

    def transferMyNet(self, path_to_trained_modelh5: str) -> Model:
        model = load_model(path_to_trained_modelh5)

        inputs = Input(shape=self.input_shape)
        x = inputs

        if self.include_top:
            for layer in model.layers[1:]:
                x = layer(x)
        else:
            for layer in model.layers[1:-4]:
                x = layer(x)

        new_model = Model(inputs=inputs, outputs=x)

        for layer in new_model.layers:
            layer.trainable = True
        
        return new_model

    def transferResnet50(self) -> Model:
        # Carregar a ResNet50 com pesos pré-treinados
        feature_extractor = ResNet50(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )

        # Definir a nova entrada
        inputs = Input(shape=self.input_shape)

        # Passar o tensor de entrada através das primeiras camadas
        x = inputs
        for layer in feature_extractor.layers:
            x = layer(x)

        # Criar o novo modelo com as camadas da ResNet50
        new_model = Model(inputs=inputs, outputs=x)

        # Tornar as camadas treináveis
        for layer in new_model.layers:
            layer.trainable = True

        return new_model

    
    def transferMobileNet(self) -> Model:
        feature_extractor = MobileNet(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )

        inputs = Input(shape=self.input_shape)
        x = inputs

        for layer in feature_extractor.layers:
            layer.trainable = True
            x = layer(x)

        return Model(inputs=inputs, outputs=x)
    
    def transferXception(self) -> Model:
        feature_extractor = Xception(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )

        inputs = Input(shape=self.input_shape)
        x = inputs

        for layer in feature_extractor.layers:
            layer.trainable = True
            x = layer(x)

        return Model(inputs=inputs, outputs=x)
    
    def transferVGG16(self) -> Model:
        feature_extractor = VGG16(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )

        inputs = Input(shape=self.input_shape)
        x = inputs

        for layer in feature_extractor.layers:
            layer.trainable = True
            x = layer(x)

        return Model(inputs=inputs, outputs=x)

def load_model(path: str):
    return keras.models.load_model(path)