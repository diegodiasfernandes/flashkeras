from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *



def minMaxScaler(x: pd.DataFrame | np.ndarray, min: float = 0, max: float = 1, return_scaler: bool = False) -> np.ndarray | tuple[MinMaxScaler, np.ndarray]:

    """
    Scale numerical features to a specified range using Min-Max normalization.

    `Flash Explanation:` *`Use this when you want to transform your data to a specific range (like [0, 1]) so that all features are comparable!`*

    Parameters
        x : pd.DataFrame | np.ndarray
            Input data to be scaled.
        min : float, default=0
            Desired minimum value of the transformed data.
        max : float, default=1
            Desired maximum value of the transformed data.
        return_scaler : bool, default=False
            If True, also returns the fitted MinMaxScaler object.

    Returns
        np.ndarray | tuple[MinMaxScaler, np.ndarray]
            Scaled data as a numpy array.  
            If `return_scaler=True`, returns a tuple containing `(scaler, scaled_data)`.
    """
    
    scaler = MinMaxScaler((min, max))
    scaler.fit(x)

    if return_scaler:
        return scaler, scaler.transform(x)
    else:
        return scaler.transform(x)

def minMaxScaleRevert(x: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    
    """
    Revert data from Min-Max scaled values back to their original scale using a fitted MinMaxScaler.

    `Flash Explanation:` *`Use this when you want to bring your normalized data back to its original range! You must already have a fitted scaler`*

    Parameters
        x : np.ndarray
            Scaled data to be inverted.
        scaler : MinMaxScaler
            Fitted MinMaxScaler instance that was originally used for scaling.

    Returns
        np.ndarray
            Data transformed back to the original scale.
    """
    
    return scaler.inverse_transform(x)
