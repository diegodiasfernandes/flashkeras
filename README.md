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
from flashkeras import FlashSequential, FlashPreProcessing, FlashDataGenerator

# Example dataset
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Image augmentation / pipeline utilities
img_gen = FlashDataGenerator(
    img_shape=(28, 28),
    rotation_range=10,
)

train_batches = img_gen.preprocess_images_from_nparray(x_train, y_train)
test_batches = img_gen.preprocess_images_from_nparray(x_test, y_test)

# Model creation
model = FlashSequential("classification")
model.addDense(128, "relu")
model.addDense(64, "relu")
model.fit(
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
from flashkeras.data_collecting import FlashDataGenerator

# Example: loading batches from directory or array
image_generator = FlashDataGenerator(img_shape=(32, 32))
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

### Notebook commands

To see the available example notebooks:

```bash
flashkeras notebooks list
```

To download a notebook locally:

```bash
flashkeras notebooks download <notebook-name>
```

Example:

```bash
flashkeras notebooks list
flashkeras notebooks download image_classification_baseline
```

This workflow is especially useful for users who want a ready-made structure to explore data, prepare examples, and run common machine learning tasks faster.

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

