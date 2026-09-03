from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *

class FlashNet:
    def __init__(self, 
                    network: Union[Sequential, Functional],
                    isFullNetwork: bool
                    ) -> None:
        """Wrap a transfer-learning network.

        `Flash Explanation:` *`Use this to record whether architecture edits are allowed.`*
        """
        self.network = network
        self.isFullNetwork: bool = isFullNetwork

class FlashTransferLearning:
    """Configure and create pretrained Keras networks for transfer learning.

    `Flash Explanation:` *`Use this class to load pretrained backbones, freeze layers, optionally truncate networks, and prepare them for reuse in another model.`*

    Parameters:
        input_shape: Shape of the input images, including height, width, and channels.
        include_top: Whether to include the pretrained classification head.
        weights: Weight source passed to the Keras application model, such as ``"imagenet"``.
        freeze: Whether to freeze all layers or the number of initial layers to freeze.
        use_only_n_layers: Whether to limit the transferred network to a number of layers.

    Attributes:
        input_shape: Configured input image shape.
        include_top: Whether the original classification head is included.
        weights: Configured pretrained weight source.
        freeze: Layer-freezing configuration.
        use_only_n_layers: Layer-truncation configuration.
    """

    def __init__(self,
                 input_shape: tuple[int, int, int] = (-1,-1,-1),
                 include_top: bool = True,
                 weights: str = "imagenet",
                 freeze: bool | int = False,
                 use_only_n_layers: bool | int = False
                 ) -> None:
        self.input_shape: tuple[int, int, int] = input_shape
        self.include_top: bool = include_top
        self.weights: str = weights
        self.freeze: bool | int = freeze
        self.use_only_n_layers: bool | int = use_only_n_layers

    def _dropLayers(self, model: Sequential) -> Sequential:
        """Keep only the requested number of layers when truncation is enabled.

        `Flash Explanation:` *`Use this internal helper to create a smaller transferable feature extractor.`*
        """
        if type(self.use_only_n_layers) == int:
            if self.use_only_n_layers == 0:
                raise ValueError(
                    "Attribute 'use_only_n_layers' can not be zero, must use at least 1 layer on Transfer Learning."
                )
            elif self.use_only_n_layers > len(model.layers):
                raise ValueError(
                    f"Attribute 'use_only_n_layers' == {self.use_only_n_layers} is higher than the maximum number of layers {len(model.layers)}."
                )
            
        if self.use_only_n_layers == False: 
            return model
        elif self.use_only_n_layers != False and self.include_top:
            print("WARNING: Skipping use_only_n_layers since include_top is True...")
            return model
            
        new_model = Sequential()
        count = 0
        for layer in model.layers:
            if count == self.use_only_n_layers:
                break
            new_model.add(layer)
            count += 1
        
        return new_model
    
    def _freezeLayers(self, model: Functional | Sequential) -> Sequential:
        """Freeze all or the first configured layers and optionally truncate the model.

        `Flash Explanation:` *`Use this internal helper to control fine-tuning scope for a pretrained network.`*
        """
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
        """Load a saved model and prepare its layers for transfer learning.

        `Flash Explanation:` *`Use this to reuse a locally trained H5 model as a feature extractor.`*

        Parameters:
            path_to_trained_modelh5: Path to the saved Keras model file.

        Returns:
            Sequential: A sequential model prepared for transfer learning.
        """
        model = keras.models.load_model(path_to_trained_modelh5)

        new_model = Sequential()

        if self.include_top:
            for layer in model.layers[1:]:
                new_model.add(layer)
        else:
            for layer in model.layers[1:]:
                if isinstance(layer, keras.layers.Flatten):
                    break
                new_model.add(layer)

        for layer in new_model.layers:
            layer.trainable = True
        
        return self._freezeLayers(new_model)

    def transferResnet50(self) -> FlashNet:
        """Create a ResNet50 transfer-learning network.

        `Flash Explanation:` *`Use this to obtain an ImageNet-backed ResNet50 with the configured freezing policy.`*

        Parameters:
            None.

        Returns:
            FlashNet: The ResNet50 network and its full-network status.

        Raises:
            ValueError: If ``use_only_n_layers`` is configured as an integer.
        """
        if type(self.use_only_n_layers) == int:
            raise ValueError(
                "ResNet can not have 'use_only_n_layers' attribute as int, use the default value (False)."
            )
        
        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearFlash() method is called.")
            model = self._freezeLayers(ResNet50(include_top=self.include_top,weights=self.weights))
            return FlashNet(model, True)
        
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

        return FlashNet(feature_extractor, False)
    
    def transferMobileNet(self) -> FlashNet:
        """Create a MobileNet transfer-learning network.

        `Flash Explanation:` *`Use this for a lightweight pretrained convolutional backbone.`*

        Parameters:
            None.

        Returns:
            FlashNet: The MobileNet network and its full-network status.

        Raises:
            ValueError: If the freeze or truncation limit exceeds the network depth.
        """
        num_layers = len(MobileNet(include_top=False).layers)
        if type(self.use_only_n_layers) == int:
            if int(self.use_only_n_layers) > num_layers:
                raise ValueError(
                    f"Attribute 'use_only_n_layers' == {self.use_only_n_layers} is higher than the maximun number of layers {num_layers}."
                )
        if type(self.freeze) == int:
            if int(self.freeze) > num_layers:
                raise ValueError(
                    f"Attribute 'freeze' == {self.freeze} is higher than the maximun number of layers {num_layers}."
                )

        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearModel() method is called.")
            model = MobileNet(include_top=self.include_top,weights=self.weights)
            return FlashNet(self._freezeLayers(model), True)
        
        feature_extractor = MobileNet(
            include_top=False,
            weights=self.weights,
            input_shape=self.input_shape
        )

        return FlashNet(self._freezeLayers(feature_extractor), False)
    
    def transferXception(self) -> FlashNet:
        """Create an Xception transfer-learning network.

        `Flash Explanation:` *`Use this to obtain an Xception backbone with the configured freezing policy.`*

        Parameters:
            None.

        Returns:
            FlashNet: The Xception network and its full-network status.

        Raises:
            ValueError: If the freeze or truncation limit exceeds the network depth.
        """
        num_layers = len(Xception(include_top=False).layers)
        if type(self.use_only_n_layers) == int:
            if int(self.use_only_n_layers) > num_layers:
                raise ValueError(
                    f"Attribute 'use_only_n_layers' == {self.use_only_n_layers} is higher than the maximun number of layers {num_layers}."
                )
        if type(self.freeze) == int:
            if int(self.freeze) > num_layers:
                raise ValueError(
                    f"Attribute 'freeze' == {self.freeze} is higher than the maximun number of layers {num_layers}."
                )
        
        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearModel() method is called.")
            model = Xception(include_top=self.include_top,weights=self.weights)
            return FlashNet(self._freezeLayers(model), True)
        
        feature_extractor = Xception(
            include_top=False,
            weights=self.weights,
            input_shape=self.input_shape
        )

        return FlashNet(self._freezeLayers(feature_extractor), False)
    
    def transferVGG16(self) -> FlashNet:
        """Create a VGG16 transfer-learning network.

        `Flash Explanation:` *`Use this to obtain a VGG16 backbone with the configured freezing policy.`*

        Parameters:
            None.

        Returns:
            FlashNet: The VGG16 network and its full-network status.

        Raises:
            ValueError: If the freeze or truncation limit exceeds the network depth.
        """
        num_layers = len(VGG16(include_top=False).layers)
        if type(self.use_only_n_layers) == int:
            if int(self.use_only_n_layers) > num_layers:
                raise ValueError(
                    f"Attribute 'use_only_n_layers' == {self.use_only_n_layers} is higher than the maximun number of layers {num_layers}."
                )
        if type(self.freeze) == int:
            if int(self.freeze) > num_layers:
                raise ValueError(
                    f"Attribute 'freeze' == {self.freeze} is higher than the maximun number of layers {num_layers}."
                )
        
        if self.include_top:
            print("WARNING: Include top == True. You can not modify this FlashKeras until the flash.clearModel() method is called.")
            model = VGG16(include_top=self.include_top,weights=self.weights)
            return FlashNet(self._freezeLayers(model), True)
        
        feature_extractor = VGG16(
            include_top=False,
            weights=self.weights,
            input_shape=self.input_shape
        )

        return FlashNet(self._freezeLayers(feature_extractor), False)

