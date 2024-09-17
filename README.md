# FlashKeras

## Installation
````
# note that this is installing tensorflow, 
# matplotlib, opencv-python, pandas and scikit-learn
pip install git+https://github.com/diegodiasfernandes/flashkeras.git 
````

## Why FlashKeras
keras is one of the best machine learning libraries, but it is super Dense *haha*. Use this to speed up your coding using pre-generated functions.

Also, FlashKeras makes use of a begginer-friendly organization that is also educative, since its modules are organized based on a machine learning pipeline (data colletion, analyses, preprocessing, model building and evaluation).

## Basic Example

Usage example:  
```py
# loading data
from flashkeras.datasets import load_mnist
(x_train, y_train), (x_test, y_test) = load_mnist()

# preprocessing
from flashkeras.preprocessing.images.FlashDataGenerator import FlashDataGenerator 
flash_gen = FlashDataGenerator (
    img_size=71, # resizing
    rotation_range=10 # rotating
)

train_batches = flash_gen.flow_classes_from_nparray(x_train, y_train)
test_batches = flash_gen.flow_classes_from_nparray(x_test, y_test)

flashkeras.preprocessing.FlashPreProcessing import FlashPreProcessing as flashprep
input_shape = flashprep.getInputShape(train_batches)

# transfer learning from MobileNet
from flashkeras.models.transferlearning.FlashTransferLearning import FlashTransferLearning
flash_transfer = FlashTransferLearning(
    input_shape=input_shape,
    include_top=False, 
    freeze=2, 
    use_only_n_layers=7    
)
xception = flash_transfer.transferXception()

# create model with FlashSequential and add the transferlearning
flash = FlashSequential()

flash.addTransferLearning(xception)
flash.add(keras.layers.Flatten())
flash.addDense(64, "relu")
flash.addDense(32, "elu")
flash.fit(train_batches=train_batches, epochs=15, validation=test_batches)
```

## Basic Pipeline and Sub-Divisions
FlashKeras is based on a basic machine learning pipeline, that being:
- Data Colletion
- Data Analysis
- Pre Processing
- Model Building
- Evaluation

So, the modules presented here are 'analysing', 'models', 'preprocessing' and 'evaluation'.

### flashkeras.analysing
- Plotting images and Graphs
- Matrixes

### flashkeras.preprocessing
#### FlashPreProcessing
- One-Hot-Encode
- Converting and reshaping
- Resizing

#### images.FlashDatGenerator
- Collect images from directory (directory batches)
- Collect images from array (np.ndarray batches)
- Applying Filters
- Resizing and reshaping

### flashkeras.models
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

#### transferlearning.FlashTransferLearning
- How many layers you want from a network
- How many frozen layers you want
- Do all of that on your own saved network

### flashkeras.evaluation
#### FlashEvaluation
- Accuracy, ...
