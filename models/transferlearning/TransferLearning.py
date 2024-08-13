from keras.applications import MobileNet, ResNet50, Xception, VGG16 # type: ignore
from keras.src.engine.functional import Functional # type: ignore
from keras.models import Sequential # type: ignore
import numpy as np # type: ignore
import keras # type: ignore

class TransferLearning:
    def __init__(self,
                 input_shape: tuple[int, int, int] = (-1,-1,-1),
                 include_top: bool = True,
                 weights: str = "imagenet",
                 freeze: bool | int = False,
                 use_n_layers: bool | int = False
                 ) -> None:
        
        self.input_shape: tuple[int, int, int] = input_shape
        self.include_top: bool = include_top
        self.weights: str = weights
        self.freeze: bool | int = freeze
        self.use_n_layers: bool | int = use_n_layers

    def _dropLayers(self, model: Sequential) -> Sequential:
        if type(self.use_n_layers) == int:
            if self.use_n_layers == 0:
                raise ValueError(
                    "Attribute 'use_n_layers' can not be zero, must use at least 1 layer on Transfer Learning."
                )
            elif self.use_n_layers > len(model.layers):
                raise ValueError(
                    f"Attribute 'use_n_layers' == {self.use_n_layers} is higher than the maximum number of layers {len(model.layers)}."
                )
        if self.use_n_layers == False: return model
            
        new_model = Sequential()
        count = 0
        for layer in model.layers:
            if count == self.use_n_layers:
                break
            new_model.add(layer)
            count += 1
        
        return new_model
    
    def _freezeLayers(self, model: Functional | Sequential) -> Sequential:
        if not self.freeze: return model

        freezed_model = model

        if type(self.freeze) == bool:
            for layer in freezed_model.layers:
                layer.trainable = False
        else:
            count = 0
            for layer in freezed_model.layers:
                if count < self.freeze: # freezing
                    layer.trainable = False
                count += 1

        return self._dropLayers(freezed_model)

    def transferMyNet(self, path_to_trained_modelh5: str) -> Sequential:
        model = keras.models.load_model(path_to_trained_modelh5)

        new_model = Sequential()

        if self.include_top:
            for layer in model.layers[1:]:
                new_model.add(layer)
        else:
            flatten_type = type(keras.layers.Flatten)
            for layer in model.layers[1:]:
                if type(layer) == flatten_type:
                    break
                new_model.add(layer)

        for layer in new_model.layers:
            layer.trainable = True
        
        return self._freezeLayers(new_model)

    def transferResnet50(self) -> tuple[Functional, bool]:
        if type(self.use_n_layers) == int:
            raise ValueError(
                "ResNet can not have 'use_n_layers' attribute as int, use the default value (False)."
            )
        
        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearFlash() method is called.")
            model = self._freezeLayers(ResNet50(include_top=self.include_top,weights=self.weights))
            return model, True
        
        feature_extractor = ResNet50(
            include_top=self.include_top,
            weights=self.weights,
            input_shape=self.input_shape
        )
        
        if self.freeze == True:
            for layer in feature_extractor.layers:
                layer.trainable = False
        elif type(self.freeze) == int:
            for layer in feature_extractor.layers[:self.freeze]:
                layer.trainable = False

        return feature_extractor, False
    
    def transferMobileNet(self) -> tuple[keras.Model, bool]:
        num_layers = len(MobileNet(include_top=False).layers)
        if type(self.use_n_layers) == int:
            if int(self.use_n_layers) > num_layers:
                raise ValueError(
                    f"Attribute 'use_n_layers' == {self.use_n_layers} is higher than the maximun number of layers {num_layers}."
                )
        if type(self.freeze) == int:
            if int(self.freeze) > num_layers:
                raise ValueError(
                    f"Attribute 'freeze' == {self.freeze} is higher than the maximun number of layers {num_layers}."
                )

        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearModel() method is called.")
            model = MobileNet(include_top=self.include_top,weights=self.weights)
            return self._freezeLayers(model), True
        
        feature_extractor = MobileNet(
            include_top=False,
            weights=self.weights,
            input_shape=self.input_shape
        )

        return self._freezeLayers(feature_extractor), False
    
    def transferXception(self) -> tuple[keras.Model, bool]:
        num_layers = len(Xception(include_top=False).layers)
        if type(self.use_n_layers) == int:
            if int(self.use_n_layers) > num_layers:
                raise ValueError(
                    f"Attribute 'use_n_layers' == {self.use_n_layers} is higher than the maximun number of layers {num_layers}."
                )
        if type(self.freeze) == int:
            if int(self.freeze) > num_layers:
                raise ValueError(
                    f"Attribute 'freeze' == {self.freeze} is higher than the maximun number of layers {num_layers}."
                )
        
        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearModel() method is called.")
            model = Xception(include_top=self.include_top,weights=self.weights)
            return self._freezeLayers(model), True
        
        feature_extractor = Xception(
            include_top=False,
            weights=self.weights,
            input_shape=self.input_shape
        )

        return self._freezeLayers(feature_extractor), False
    
    def transferVGG16(self) -> tuple[keras.Model, bool]:
        num_layers = len(VGG16(include_top=False).layers)
        if type(self.use_n_layers) == int:
            if int(self.use_n_layers) > num_layers:
                raise ValueError(
                    f"Attribute 'use_n_layers' == {self.use_n_layers} is higher than the maximun number of layers {num_layers}."
                )
        if type(self.freeze) == int:
            if int(self.freeze) > num_layers:
                raise ValueError(
                    f"Attribute 'freeze' == {self.freeze} is higher than the maximun number of layers {num_layers}."
                )
        
        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearModel() method is called.")
            model = VGG16(include_top=self.include_top,weights=self.weights)
            return self._freezeLayers(model), True
        
        feature_extractor = VGG16(
            include_top=False,
            weights=self.weights,
            input_shape=self.input_shape
        )

        return self._freezeLayers(feature_extractor), False

