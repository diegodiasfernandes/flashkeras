# flashkeras
keras is one of the best machine learning libraries, but it is super Dense *haha*. Use this to speed up your coding, or even consulting, as some of its methods put together keras features that doesn't seem to be related but have to be...

Also, since often keras is not used by itself, flashkeras uses also: scikit-learn, pandas, numpy, matplotlib, ...

## Basic Example

Usage example:  
```py

# loading data
(x_train, y_train), (x_test, y_test) = load_mnist_dataset()

# preprocessing
flash_gen = FlashDataGenerator (
    img_size=224, # resizing
    rotation_range=10 # rotating
)

train_batches = flash_gen.flow_classes_from_nparray(x_train, y_train)
test_batches = flash_gen.flow_classes_from_nparray(x_test, y_test)

# transfer learning from MobileNet
flash_transfer = FlashTransferLearning(
    input_shape=(224, 224, 3),
    include_top=False, 
    freeze=2, 
    use_only_n_layers=7    
)
mobilenet = flash_transfer.transferMobileNet()

# create model with FlashSequential and add the transferlearning
flash = FlashSequential()

flash.addTransferLearning(mobilenet)
flash.add(keras.layers.Flatten())
flash.addDense(64, "relu")
flash.addDense(32, "elu")
flash.fit(train_batches=train_batches, epochs=15, validation=test_batches)
```

## Basic Pipeline and Sub-Divisions
FlashKeras is based on a basic machine learning pipeline, that being:
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