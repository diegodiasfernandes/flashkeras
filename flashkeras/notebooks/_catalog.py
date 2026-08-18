"""
Registry of notebook templates shipped with flashkeras.

To add a new notebook:
    1. Create the .ipynb file in flashkeras/notebooks/templates/
       (the first code cell should be tagged "parameters" — see existing
       notebooks for the convention).
    2. Add an entry below.
    3. Add the filename to package_data in setup.py if it isn't already
       covered by the templates/*.ipynb glob.
"""

NOTEBOOKS = {
    "eda_dataframe": {
        "title": "Exploratory Data Analysis (DataFrame)",
        "description": (
            "Default EDA for a pandas DataFrame loaded from a CSV: shape, dtypes, "
            "missing values, duplicates, distributions, value counts, and a "
            "correlation matrix."
        ),
        "filename": "eda_dataframe.ipynb",
        "tags": ["eda", "pandas", "tabular"],
        "parameters": {
            "csv_path": "Path to your CSV file.",
            "target_column": "Optional. Name of the column you want to predict.",
            "separator": "CSV delimiter (default ',').",
        },
    },
    "image_classification_baseline": {
        "title": "Image Classification Baseline",
        "description": (
            "Keras CNN baseline for image classification: loads images from a "
            "directory structure, applies augmentation, trains a small CNN with "
            "early stopping, and plots training curves."
        ),
        "filename": "image_classification_baseline.ipynb",
        "tags": ["cv", "keras", "classification", "images"],
        "parameters": {
            "data_dir": "Path to a directory with one subfolder per class.",
            "image_size": "Target (height, width) for resizing images.",
            "batch_size": "Training batch size.",
            "validation_split": "Fraction of data held out for validation.",
            "epochs": "Number of training epochs.",
        },
    },
    "text_classification_baseline": {
        "title": "Text Classification Baseline",
        "description": (
            "Keras baseline for text classification from a CSV: TextVectorization, "
            "an embedding + pooling model, training with early stopping, and "
            "training curves."
        ),
        "filename": "text_classification_baseline.ipynb",
        "tags": ["nlp", "keras", "classification", "text"],
        "parameters": {
            "csv_path": "Path to your CSV file.",
            "text_column": "Name of the column containing text.",
            "label_column": "Name of the column containing labels.",
            "max_tokens": "Vocabulary size for TextVectorization.",
            "sequence_length": "Fixed sequence length after vectorization.",
        },
    },
}
