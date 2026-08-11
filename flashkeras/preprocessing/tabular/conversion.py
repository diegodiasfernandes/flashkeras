from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *


def datasetPandasToNumpyNDArray(
        data: Union[pd.Series, pd.DataFrame], 
    ) -> np.ndarray:
    """
    Converts a Pandas Series or DataFrame into a NumPy array.

    `Flash Explanation:` *`Use this when you have a Pandas DataFrame or Series and need to transform it into a NumPy array!`*  

    This function leverages the built-in `.to_numpy()` method from Pandas, 
    ensuring efficient conversion of tabular or single-column data into 
    NumPy format.  

    Parameters:
        data : pd.Series | pd.DataFrame  
            The Pandas object containing the data to convert.  

    Returns:
        np.ndarray  
            A NumPy array containing the same data.  
    """
    return data.to_numpy()
        
