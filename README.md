# FlashKeras

FlashKeras is a lightweight Python toolkit built on top of Keras to accelerate common deep learning and ML workflows, especially around image processing, preprocessing, model creation, and evaluation.

The project was designed to reduce boilerplate and keep the workflow close to a clear machine learning pipeline: data collection, analysis, preprocessing, model building, and evaluation.

## Why use FlashKeras?

Keras is powerful, but it can feel verbose for everyday experiments and quick prototypes. FlashKeras focuses on:

- Faster setup for standard ML workflows
- Cleaner organization by pipeline stage
- Simple abstractions for common tasks
- Practical utilities for image data and tabular data
- Compatibility with TensorFlow/Keras workflows

## Main idea

Instead of forcing users to manually assemble each part of a pipeline, FlashKeras organizes the library into modular blocks that reflect how real ML projects are usually structured:

- `flashkeras.data_collecting` — data import and dataset utilities
- `flashkeras.analysing` — visualization and inspection helpers
- `flashkeras.preprocessing` — normalization, encoding, reshaping, resizing
- `flashkeras.models` — simplified sequential modeling and transfer learning
- `flashkeras.evaluation` — metrics and evaluation tools
- `flashkeras.utils` — reusable helpers and internal utilities

This keeps the library easy to understand, easy to extend, and aligned with a practical ML workflow.

## Installation

Install from PyPI:

```bash
pip install flashkeras
```

Install directly from GitHub:

```bash
pip install git+https://github.com/diegodiasfernandes/flashkeras.git
```

This package depends on:

- TensorFlow
- Matplotlib
- OpenCV
- pandas
- scikit-learn

## Quick start

```python
from flashkeras import FlashSequential
from flashkeras.data_collecting import load_mnist
from flashkeras.preprocessing import preprocess_images_from_nparray

# Example dataset
(x_train, y_train), (x_test, y_test) = load_mnist()

# Preprocessing
# Image augmentation / pipeline utilities
train_batches = preprocess_images_from_nparray(
    x_train,
    y_train,
    img_shape=(28, 28),
    rotation_range=10,
)
test_batches = preprocess_images_from_nparray(
    x_test,
    y_test,
    img_shape=(28, 28),
)

# Model creation
model = FlashSequential("classification")
model.addDense(128, "relu")
model.addDense(64, "relu")
model.train(
    x=train_batches,
    epochs=10,
    validation_data=test_batches,
    auto_output_layer=True
)

# Evaluation
from flashkeras.evaluation import getAccuracy, getRecall

acc = getAccuracy(model, x_test, y_test)
recall = getRecall(model, x_test, y_test)
print(acc, recall)
```

## Core features

### 1) Data collection
Utilities to work with datasets and image sources in a simple way.

```python
from flashkeras.data_collecting import load_all_classes_from_directory_and_preprocess

# Example: loading batches from directory or array
image_batches = load_all_classes_from_directory_and_preprocess(
    "path/to/dataset",
    img_shape=(32, 32),
)
```

### 2) Analysis and visualization
Helpful functions for inspecting data and model behavior visually.

```python
from flashkeras.analysing import show_images_nparray

show_images_nparray(x_train, num_images=5)
```

### 3) Preprocessing
This is one of the strongest areas of the package. It includes utilities for:

- train/test splitting
- label encoding and decoding
- resizing and reshaping
- normalization
- image format conversion
- tabular feature preparation

```python
from flashkeras import FlashPreProcessing

labels = ["cat", "dog", "cat", "fish"]
encoded, encoder = FlashPreProcessing.labelEncoder(labels, True)
print(encoded)
print(FlashPreProcessing.labelDecoder([2], encoder))
```

### 4) Model building
The library exposes simplified model abstractions for fast experimentation.

```python
from flashkeras import FlashSequential

flash_model = FlashSequential("classification")
flash_model.addDense(32, "relu")
flash_model.addDense(10, "softmax")
```

This wrapper keeps the workflow close to Keras, while reducing repetitive setup work for common tasks.

## How to Use FlashSequential

`FlashSequential` is a wrapper around `keras.Sequential` for building models in a linear order: each layer receives the output of the previous layer. It keeps Keras flexibility while providing short methods for common operations, such as adding dense layers, converting images into vectors, training, and making predictions.

### Dense and Fully Connected Layers

In Keras, `Dense` is the implementation of a **fully connected layer**. Each unit in the layer receives input from every unit in the previous layer and computes a weighted combination, usually followed by an activation function.

The concepts below describe the same idea at different levels:

| Concept | Meaning | How it appears in FlashKeras |
| --- | --- | --- |
| Fully connected layer | An architectural concept in which each unit connects to every input. | Describes the role of the layer. |
| `Dense` | Keras implementation of this concept. | `model.addDense(64, "relu")` or `model.add(Dense(64, activation="relu"))` |
| Unit/neuron | A unit that produces one value inside a `Dense` layer. | The first `addDense` argument, such as `64`, defines the number of units. |
| Activation | A function applied to the layer output, such as `relu`, `sigmoid`, or `softmax`. | The second `addDense` argument defines the activation. |
| Output layer | The final layer responsible for producing predictions. | It can be added automatically with `auto_output_layer=True`. |

### Why Use Flatten with Images?

An image usually reaches the model as a tensor with the shape `(height, width, channels)`, for example `(28, 28, 1)` for a grayscale image. An architecture based only on `Dense` layers is easier to reason about when each example is represented as a vector.

`Flatten` rearranges the values without learning parameters: `(28, 28, 1)` becomes `(784,)`. It does not resize the image, normalize its pixels, or reduce the amount of information; it only removes the spatial structure so a fully connected layer can receive all pixels as inputs.

```python
from flashkeras import FlashSequential

model = FlashSequential("classification")
model.addFlatten()            # (28, 28, 1) -> (784,)
model.addDense(128, "relu")   # hidden layer

# The output layer is inferred from y_train.
model.train(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    auto_output_layer=True,
)
```

Use `addFlatten()` when the next part of the model is a `Dense` layer and the data still has spatial dimensions. In a convolutional model, it usually appears after `Conv2D` and pooling layers, before the dense layers:

```python
from flashkeras import FlashSequential
from tensorflow.keras.layers import MaxPooling2D

model = FlashSequential("classification")
model.addConvolution2D(32, (3, 3), activation="relu")
model.add(MaxPooling2D())
model.addFlatten()
model.addDense(64, "relu")
model.train(x=x_train, y=y_train, epochs=5, auto_output_layer=True)
```

`Flatten` is not required for every model. Convolutional networks can use `GlobalAveragePooling2D`, and architectures that already produce vectors can go directly to `Dense`. You should also not use `Flatten` before `Conv2D`, because convolution needs the image height and width to remain structured.

### Complete Workflow

The recommended workflow is to build the architecture, inspect the summary, and train. `train()` automatically configures the input shape, compiles the model when necessary, and can create the task-appropriate output layer.

```python
from flashkeras import FlashSequential

model = FlashSequential("classification")
model.addFlatten()
model.addDense(128, "relu")
model.addDense(64, "relu")

model.summary()
history = model.train(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    auto_output_layer=True,
)

predictions = model.predict(x_test)
one_prediction = model.singlePredict(x_test[0])
```

| Situation | Recommended structure | Notes |
| --- | --- | --- |
| Tabular data | `Dense -> Dense -> output` | Each sample is already a feature vector. |
| Image with `Dense` only | `Flatten -> Dense -> output` | Converts the image tensor into a vector. |
| Image with convolution | `Conv2D -> Pooling -> Flatten -> Dense -> output` | Preserves spatial structure during feature extraction. |
| Binary classification | Hidden layers + automatic output | FlashKeras infers a one-unit `sigmoid` output. |
| Multiclass classification | Hidden layers + automatic output | FlashKeras infers `softmax` and the number of classes from the labels. |
| Regression | Hidden layers + automatic output | FlashKeras uses a linear output and `mse` as the default loss. |

The `addDense()` and `addFlatten()` methods are shortcuts. For any Keras-compatible layer, use `model.add(layer)`. After a complete model is loaded with `loadModel()`, architecture changes are blocked to preserve the restored model.

### 5) Transfer learning
The library includes helpers for transfer learning patterns, enabling reuse of pretrained networks with a simpler API.

```python
from flashkeras.models import FlashTransferLearning

transfer = FlashTransferLearning(
    input_shape=(224, 224, 3),
    include_top=False,
    freeze=2,
    use_only_n_layers=7
)
```

### 6) Evaluation
The evaluation module provides metrics for model quality analysis.

```python
from flashkeras.evaluation import getAccuracy, getPrecision, getRecall

accuracy = getAccuracy(model, x_test, y_test)
precision = getPrecision(model, x_test, y_test)
recall = getRecall(model, x_test, y_test)
```

## Pipeline-based architecture

FlashKeras is intentionally organized around the standard ML development lifecycle:

1. Data collection
2. Analysis
3. Preprocessing
4. Model creation
5. Evaluation

This modular structure makes it easier for users to:

- understand each step of the pipeline
- reuse only the modules they need
- apply the library to experiments and prototypes quickly
- keep code readable and educational

## Example notebooks

FlashKeras also includes notebook-based templates to help users start working quickly on recurring tasks without writing everything from scratch. These notebooks are designed as practical starting points for common workflows, especially in exploratory analysis and image classification.

Examples included in the project:

- exploratory data analysis for tabular data
- image classification baseline workflows
- visual inspection of datasets and model outputs
- experimentation templates for preprocessing + training loops

This makes the library useful not only as a Python package, but also as a teaching and prototyping tool for structured ML work.

### Generate notebooks from the command line

FlashKeras can generate ready-to-edit Jupyter notebooks directly from the Windows
Command Prompt. This is useful for users who are unsure where to start: the
catalog and descriptions help identify the right workflow. It also saves
experienced users from repeatedly creating boilerplate for common experiments.

Open **cmd**, activate the environment where FlashKeras is installed, and list
the available templates:

```cmd
flashkeras notebooks list
```

Filter the catalog by a topic when you already know the kind of task you need:

```cmd
flashkeras notebooks list --tag tabular
flashkeras notebooks list --tag cv
flashkeras notebooks list --tag evaluation
```

Inspect a template before generating it. The command shows its purpose,
parameters, and output filename:

```cmd
flashkeras notebooks describe image_classification_baseline
```

Create a notebook in the current directory with `notebooks new`:

```cmd
flashkeras notebooks new image_classification_baseline
```

Choose a destination directory or filename when organizing several experiments:

```cmd
flashkeras notebooks new eda_dataframe --dest notebooks
flashkeras notebooks new text_classification_baseline --dest experiments\text_baseline.ipynb
```

The available template names currently include:

- `eda_dataframe` for exploratory analysis of tabular data
- `dataframe_classification_validation` for validating a DataFrame classifier
- `image_classification_baseline` for a starter CNN image-classification workflow
- `image_classification_validation` for evaluating an image classifier
- `text_classification_baseline` for a starter text-classification workflow

If the destination file already exists, FlashKeras keeps it unchanged unless
you explicitly allow replacement:

```cmd
flashkeras notebooks new image_classification_baseline --dest experiments\baseline.ipynb --overwrite
```

After generation, open the `.ipynb` file in Jupyter or VS Code and replace the
template paths and parameters with your own data. The generated notebook is a
starting point, so it can be adapted without changing the original template.

## Project focus

FlashKeras is not trying to replace Keras or become a full general-purpose framework. Its focus is to simplify the most common deep learning tasks and reduce boilerplate, especially for:

- quick experiments
- educational projects
- image classification pipelines
- preprocessing-heavy workflows
- beginner-friendly Keras usage

## Contributing

Contributions are welcome. If you want to improve the library, add utilities, or improve documentation, feel free to open a pull request.

## License

This project is distributed under the MIT license.

## Project status

FlashKeras is intended as a practical, compact toolkit for rapid prototyping and educational ML workflows. It prioritizes simplicity, modular organization, and ease of use without losing compatibility with the underlying Keras ecosystem.

