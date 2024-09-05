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

### analysing
- Plotting images and Graphs
- Matrixes

### preprocessing
#### Class FlashPreProcessing
- One-Hot-Encode
- Converting and reshaping

#### preprocessing.images.FlashDatGenerator
- Collect images from directory
- Filters