from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *


def ensureOneHotEncoding(
        y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:

    """
    Ensures that a label array or Series is converted to one-hot encoding.

    `Flash Explanation:` *`Use this when you want to guarantee your labels are in one-hot format!`*  

    This function accepts either a NumPy array or a Pandas Series. If a Series is provided, 
    it first encodes categories as integer class codes, then converts them to one-hot.  
    If the input is already in one-hot format (2D with more than one column), it is returned unchanged.  
    Otherwise, raw integer labels are transformed into one-hot vectors.  

    Parameters:
        y: np.ndarray | pd.Series  
            The labels to be encoded. Can be:  
            - A Pandas Series of categorical values.  
            - A NumPy array of integer labels.  
            - A NumPy array already in one-hot encoded format.  

    Returns:
        np.ndarray  
            The labels in one-hot encoded format with shape (n_samples, n_classes).  

    Examples:
    --------
    >>> import numpy as np, pandas as pd
    >>> from tensorflow.keras.utils import to_categorical
    >>> from my_module import ensureOneHotEncoding

    #### Convert Pandas categorical Series to one-hot
    >>> y_series = pd.Series(["cat", "dog", "dog", "cat"])
    >>> encoded = ensureOneHotEncoding(y_series)
    >>> print(encoded)
    [[1. 0.]
        [0. 1.]
        [0. 1.]
        [1. 0.]]

    ---
    #### Convert integer NumPy array to one-hot
    >>> y_array = np.array([0, 1, 2])
    >>> encoded = ensureOneHotEncoding(y_array)
    >>> print(encoded)
    [[1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]]

    ---
    #### Pass-through when already one-hot
    >>> y_onehot = np.array([[1, 0], [0, 1], [1, 0]])
    >>> encoded = ensureOneHotEncoding(y_onehot)
    >>> print(np.allclose(encoded, y_onehot))
    True
    """

    if isinstance(y, pd.Series):
        arr = y.astype('category').cat.codes
        return to_categorical(arr)

    if len(y.shape) > 1 and y.shape[1] > 1:
        return y

    return to_categorical(y)

def labelEncoder(
        y: Union[np.ndarray, pd.Series, list], 
        return_encoder: bool = False
    ) -> np.ndarray | tuple[np.ndarray, LabelEncoder]:
    
    """
    Encode categorical labels into numerical format using scikit-learn's LabelEncoder.

    `Flash Explanation:` *`Use this when you need to convert class labels (like 'cat', 'dog') into numbers (like 0, 1) for model training!`*

    ## Parameters
        y : np.ndarray | pd.Series | list
            Array-like object with categorical labels.
        
        return_encoder : bool, default=False
            If True, also returns the fitted LabelEncoder object.

    ## Returns
        np.ndarray | tuple[np.ndarray, LabelEncoder]
            Encoded labels as a numpy array.  
            If `return_encoder=True`, returns a tuple containing `(encoded_labels, encoder)`.
    """
    
    le: LabelEncoder = LabelEncoder()
    le.fit(y)

    if return_encoder:
        return le.transform(y), le

    return le.transform(y)

def labelDecoder(
            labels: Union[np.ndarray, pd.Series, list], 
            encoder: LabelEncoder
        ) -> np.ndarray:

        """
        Decode numerical labels back into their original categorical form using a fitted LabelEncoder.

        `Flash Explanation:` *`Use this when you want to transform your encoded labels (like 0, 1) back into the original categories (like 'cat', 'dog')! 
        You must already have a fitted encoder.`*

        Parameters
            labels : np.ndarray | pd.Series | list
                Encoded labels to be transformed back.
            encoder : LabelEncoder
                Fitted LabelEncoder instance used for the original encoding.

        Returns
            np.ndarray
                Decoded labels in their original categorical form.
        """

        return (encoder.inverse_transform(labels))