# FlashKeras

## Installation
````
pip install git+https://github.com/diegodiasfernandes/flashkeras.git 
````
Note that `pip install git+https://github.com/diegodiasfernandes/flashkeras.git` would also install `tensorflow`, `matplotlib`, `opencv-python`, `pandas` and ``scikit-learn``.

## Why FlashKeras
keras is one of the best machine learning libraries, but it is super Dense *haha*. Use this to speed up your coding using pre-generated functions.

Also, FlashKeras makes use of a begginer-friendly organization that is also educative, since its modules are organized based on a machine learning pipeline (data colletion, analyses, preprocessing, model building and evaluation).

## Basic Example

Usage example:  
```py
# 1) collecting data
from flashkeras.data_collecting import load_mnist
(x_train, y_train), (x_test, y_test) = load_mnist()

# 2) analysing
from flashkeras.analysing import show_images_nparray
show_images_nparray(x_train, num_images=5)

# 3) preprocessing
from flashkeras.preprocessing import FlashDataGenerator 
flash_gen = FlashDataGenerator (
    img_size=32, # resizing
    rotation_range=10 # rotating
)

train_batches = flash_gen.flow_classes_from_nparray(x_train, y_train)
test_batches = flash_gen.flow_classes_from_nparray(x_test, y_test)

from flashkeras.preprocessing import FlashPreProcessing as flashprep
input_shape = flashprep.getInputShape(train_batches)

# 4) model building
from flashkeras.models import FlashSequential
flash = FlashSequential()

flash.addTransferLearning(xception)
flash.add(Flatten())
flash.add(keras.layers.Dense(64, activation="relu")) # It is also compatible with keras!
flash.add(Dense(32, activation="elu"))
flash.fit(train_batches=train_batches, epochs=15, validation=test_batches)

# 5) evaluating
# in-development...
```

## Basic Pipeline and Sub-Divisions
FlashKeras is based on a basic machine learning pipeline, that being:
- Data Colletion
- Data Analysis
- Pre Processing
- Model Building
- Evaluation

So, the modules presented here are 'analysing', 'models', 'preprocessing' and 'evaluation'.

### ``flashkeras.analysing``
- Plotting images and Graphs
- Matrixes

### ``flashkeras.preprocessing``
#### FlashPreProcessing
- One-Hot-Encode
- Converting and reshaping
- Resizing

#### ``images.FlashDataGenerator``
- Collect images from directory (directory batches)
- Collect images from array (np.ndarray batches)
- Applying Filters
- Resizing and reshaping

### ``flashkeras.models``
#### FlashSequential
- Easier to use Sequential model.

Example:  
```py
flash = FlashSequential()
flash.addDense(32, "relu")
flash.fit(x_train, y_train, epochs=15, validation=(x_test, y_test))
```
*What's new???* there is no need to (besides possible...):
- Give the input_shape
- Create the output layer
- Set the Optimizer
- And many more

**TIP:** Also, at any point (before fit of course), if you want to use one of the many other most specific keras functions you can use flash.getSequential() and continue from there!

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
from flashkeras.models.transferlearning import FlashTransferLearning
flash_transfer = FlashTransferLearning(
    input_shape=input_shape,
    include_top=False, # excluding the Dense architecture
    freeze=2, # freezing the first 2 layers
    use_only_n_layers=7 # using only the first 7 layers
)
mobile_net = flash_transfer.transferMobileNet()
```
### flashkeras.evaluation
#### FlashEvaluation
- Accuracy, ...
