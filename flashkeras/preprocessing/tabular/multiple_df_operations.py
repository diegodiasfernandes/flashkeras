from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *



def stackDataFrames(
        matrix_a: pd.DataFrame | pd.Series | np.ndarray, 
        matrix_b: pd.DataFrame | pd.Series | np.ndarray
    ) -> pd.DataFrame | pd.Series | np.ndarray:
    
    """
    Stacks two data structures (DataFrames, Series, or NumPy arrays) vertically 
    while preserving their type and structure.
    
    `Flash Explanation:` *`Use this when you want to combine two DataFrames, Series, or NumPy arrays into one, making sure they stay consistent in format!`*  
    This function converts both inputs into 2D arrays, verifies they have the same 
    number of columns, stacks them row-wise, and returns the result in the 
    appropriate format (DataFrame, Series, or NumPy array) based on the inputs.

    Parameters:
        matrix_a: pd.DataFrame | pd.Series | np.ndarray  
            The first data structure to be stacked. Can be a pandas DataFrame, 
            Series, or NumPy array.  
        matrix_b: pd.DataFrame | pd.Series | np.ndarray  
            The second data structure to be stacked. Must be of the same type 
            as `matrix_a` and have the same number of columns (if 2D).

    Returns:
        pd.DataFrame | pd.Series | np.ndarray  
            - If both inputs are `pd.DataFrame`: returns a new DataFrame with 
            combined rows and original column names.  
            - If both inputs are `pd.Series`: returns a new Series with combined 
            values and the same name as the original Series.  
            - If both inputs are `np.ndarray`: returns a stacked NumPy array. 
            1D arrays are flattened back after stacking.  
            - If inputs differ in type: returns a NumPy array.  

    Raises:
        ValueError:  
            - If the inputs are not `pd.DataFrame`, `pd.Series`, or `np.ndarray`.  
            - If the number of columns in the two inputs do not match.  

    ---
    Examples:
    >>> import pandas as pd
    >>> import numpy as np
    >>> from my_module import stackDataFrames

    #### Combine DataFrames
    >>> df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    >>> df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    >>> result = stackDataFrames(df1, df2)
    >>> print(result)
        a  b
    0  1  3
    1  2  4
    2  5  7
    3  6  8

    ---
    #### Combine Series
    >>> s1 = pd.Series([1, 2, 3], name="x")
    >>> s2 = pd.Series([4, 5], name="x")
    >>> result = stackDataFrames(s1, s2)
    >>> print(result)
    0    1
    1    2
    2    3
    3    4
    4    5
    Name: x, dtype: int64

    ---
    #### Combine NumPy arrays
    >>> a1 = np.array([[1, 2], [3, 4]])
    >>> a2 = np.array([[5, 6]])
    >>> result = stackDataFrames(a1, a2)
    >>> print(result)
    [[1 2]
        [3 4]
        [5 6]]
    """

    
    def to_2d_array(data):
        """Convert a supported tabular value to a two-dimensional array.

        `Flash Explanation:` *`Use this internal helper to normalize input shapes before concatenation.`*
        """
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return data.reshape(-1, 1)
            return data
        else:
            raise ValueError("Inputs must be either pd.DataFrame, pd.Series, or np.ndarray")

    matrix_a_2d = to_2d_array(matrix_a)
    matrix_b_2d = to_2d_array(matrix_b)

    if matrix_a_2d.shape[1] != matrix_b_2d.shape[1]:
        raise ValueError(f"Shape mismatch: matrix_a has shape {matrix_a_2d.shape} and matrix_b has shape {matrix_b_2d.shape}. Both must have the same number of columns.")

    stacked = np.vstack((matrix_a_2d, matrix_b_2d))

    if isinstance(matrix_a, pd.DataFrame) and isinstance(matrix_b, pd.DataFrame):
        return pd.DataFrame(stacked, columns=matrix_a.columns)
    
    elif isinstance(matrix_a, pd.Series) and isinstance(matrix_b, pd.Series):
        return pd.Series(stacked.flatten(), name=matrix_a.name)
    
    elif isinstance(matrix_a, np.ndarray) and isinstance(matrix_b, np.ndarray):
        if matrix_a.ndim == 1 and matrix_b.ndim == 1:
            return stacked.flatten()
        return stacked 
    
    return stacked

