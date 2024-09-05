# flashkeras
keras is one of the best machine learning libraries, but it is super Dense *haha*. Use this to speed up your coding, or even consulting, as some of its methods put together keras features that doesn't seem to be related but have to be...

Also, since often keras is not used by itself, flashkeras uses also: scikit-learn, pandas, numpy, matplotlib, ...

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
    Usage example:  
        flash = FlashSequential()
        flash.addDense(32, "relu")
        flash.fit(x_train, y_train, epochs=15, validation=(x_test, y_test))

    TIP: Also, at any point (before fit of course), if you want to use one of the many other most specific keras functions you can use flash.getSequential() and continue from there!

#### transferlearning.FlashTransferLearning
- How many layers you want from a network
- How many frozen layers you want
- Do all of that on your own saved network

### flashkeras.evaluation
#### FlashEvaluation
- Accuracy, ...