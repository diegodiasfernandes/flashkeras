from flashkeras.utils.kerasimports import Sequential

def print_model_summary(model: Sequential):
    # Header
    print(f"{'Index':<5} | {'Layer Type':<24} {'(trainable)':<11} | Configuration")
    print("-" * 80)

    # Iterate through model layers
    for idx, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        config = layer.get_config()
        trainable = "(True)" if layer.trainable else "(freezed)"

        if layer_type == 'Dense':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Units: {config['units']}, Activation: {config['activation']}")

        elif layer_type == 'Conv2D':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Filters: {config['filters']}, Kernel Size: {config['kernel_size']}, Activation: {config['activation']}")

        elif layer_type == 'DepthwiseConv2D':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Depth Multiplier: {config['depth_multiplier']}, Kernel Size: {config['kernel_size']}, Activation: {config['activation']}")

        elif layer_type == 'BatchNormalization':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Epsilon: {config['epsilon']}, Momentum: {config['momentum']}")

        elif layer_type == 'Dropout':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Rate: {config['rate']}")

        elif layer_type == 'ReLU' or 'activation' in config:
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Activation: {layer_type.lower()}")

        elif layer_type == 'Flatten':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | None ")

        elif layer_type == 'MaxPooling2D':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Pool Size: {config['pool_size']}")

        elif layer_type == 'AveragePooling2D':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Pool Size: {config['pool_size']}")

        elif layer_type == 'GlobalMaxPooling2D':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Global Max Pooling")

        elif layer_type == 'GlobalAveragePooling2D':
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | Global Average Pooling")

        else:
            print(f"{idx:<5} | {layer_type:<25}{trainable:<11} | --- ")