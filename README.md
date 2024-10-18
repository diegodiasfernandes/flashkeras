# FlashKeras

## Installation
Download it directly from PyPi:
````
pip install flashkeras
````
Download it from GitHub:
````
pip install git+https://github.com/diegodiasfernandes/flashkeras.git 
````
Note that it would also install `tensorflow`, `matplotlib`, `opencv-python`, `pandas` and ``scikit-learn``.

## Why FlashKeras
keras is one of the best machine learning libraries, but it is super *Dense*. So, ``FlashKeras`` was made to speed up your coding using pre-generated functions.

Also, FlashKeras makes use of a begginer-friendly organization that is also educative, since its modules are organized based on a machine learning pipeline (data colletion, analyses, preprocessing, model building and evaluation).

## Basic Example

Usage example:  
```py
# 1) collecting data
from flashkeras.data_collecting.datasets import load_mnist
(x_train, y_train), (x_test, y_test) = load_mnist()

# 2) analysing
from flashkeras.analysing import show_images_nparray
show_images_nparray(x_train, num_images=5)

# 3) preprocessing
from flashkeras.preprocessing import FlashDataGenerator 
flash_gen = FlashDataGenerator (
    img_shape=(32, 32), # resizing
    rotation_range=10 # rotating
)

train_batches = flash_gen.flow_classes_from_nparray(x_train, y_train)
test_batches = flash_gen.flow_classes_from_nparray(x_test, y_test)

# 4) model building
from flashkeras.models.layers import *
from flashkeras.models import FlashSequential
flash = FlashSequential('classification')
flash.add(Flatten())
flash.add(keras.layers.Dense(64, activation="relu")) # It is also compatible with keras!
flash.add(Dense(32, activation="elu"))
flash.fit(train_batches=train_batches, epochs=15, validation=test_batches, auto_output_layer=True)

# 5) evaluating
from flashkeras.evaluation import FlashEvaluating as eval
recall = eval.getRecall(flash, x_test, y_test)
```

## Pipeline and Sub-Divisions
FlashKeras is based on a basic machine learning pipeline, that being:
- Data Colletion
- Data Analysis
- Pre Processing
- Model Building
- Evaluation

So, the modules presented here are 'analysing', 'models', 'preprocessing' and 'evaluation'.

### 1) Data Collection: ``flashkeras.data_collecting``
- Get images from directories
- Get numpy array images
- Get datasets from keras and sklearn s.a. iris, MNIST, ...

```py
from flashkeras.data_collecting import FlashDataGenerator as datagen
from flashkeras.preprocessing import FlashPreProcessing as prep

# In Order to use DirectoryIterators you must have directories inside the main dir
img_shape = prep.getImageShape('data\\myImages\\image1.png')
flash_gen = datagen(img_shape)
image_batches = flash_gen.flow_images_from_directory('data')
```

### 2) Analyses: ``flashkeras.analysing``
- Plotting images and Graphs
- Matrixes

```py
from flashkeras.analysing.graphs.line_graphs import plot_multi_line_graph

y1 = [0.1, 0.5, 0.8, 0.7, 0.9]
y2 = [0.2, 0.4, 0.6, 0.9, 1.0]
y3 = [0.3, 0.6, 0.4, 0.5, 0.8]

plot_multi_line_graph(
    [y1, y2, y3], 
    graph_title='Multiple Line Graph Example', 
    line_labels=['Real Data 1', 'Real Data 2', 'Real Data 3']
)
```

![Multi-Line Graph](https://imgur.com/vqJPpjY.png)

### 3) Preprocessing: ``flashkeras.preprocessing``
#### FlashPreProcessing
- One-Hot-Encode
- Converting and reshaping
- Resizing

```py
>>> from flashkeras.preprocessing import FlashPreProcessing as prep

>>> classes = ['dog', 'dog', 'fish', 'dog' 'cat']

>>> classes_encoded, encoder = prep.labelEncoder(classes, True)
>>> classes_encoded
[0 0 2 1]

>>> class_decoded = prep.labelDecoder([2], encoder)
>>> class_decoded
['fish']
```

#### ``images.FlashDataGenerator``
- Collect images from directory (directory batches)
- Collect images from array (np.ndarray batches)
- Applying Filters
- Resizing and reshaping

### 4) Model Building: ``flashkeras.models``
#### FlashSequential
- Easier to use Sequential model.

Example:  
```py
flash = FlashSequential('classification')
flash.addDense(32, "relu")
flash.fit(x_train, y_train, epochs=15, validation=(x_test, y_test), auto_output_layer=True)
```
*What's new???* there is no need to (besides possible...):
- Give the input_shape
- Create the output layer
- Set the Optimizer
- And many more

**TIP:** Also, at any point (before fit of course), if you want to use one of the many other most specific keras functions you can always use FlashSequential.model as a normal keras.Sequential model!

#### ``transferlearning.FlashTransferLearning``
- How many layers you want from a network
- How many frozen layers you want
- Do all of that on your own saved network

##### Example
```python
# example of input_shape from flashkeras
from flashkeras.preprocessing import FlashPreProcessing as flashprep
input_shape = flashprep.getInputShape(train_batches)

# transfer learning from MobileNet
from flashkeras.models import FlashTransferLearning
flash_transfer = FlashTransferLearning(
    input_shape=input_shape,
    include_top=False, # excluding the Dense architecture
    freeze=2, # freezing the first 2 layers
    use_only_n_layers=7 # using only the first 7 layers
)
mobile_net = flash_transfer.transferMobileNet()

# add mobilenet to the flash sequential
flash.addTransferLearning(mobile_net)
```
### 5) Evaluation: flashkeras.evaluation
#### FlashEvaluation
- Accuracy, Recall, Precision, F1-Score
- MSE, MEA

```py
from flashkeras.evaluation import FlashEvaluating as eval
recall = eval.getRecall(flash, test_batches)
```
