from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *
from flashkeras.models import FlashSequential 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, f1_score, roc_curve # type: ignore

def _adjustClassMetrics(model, 
                        x_test: pd.DataFrame | np.ndarray | Any, 
                        y_test: pd.Series | np.ndarray | None = None
                        ) -> tuple[pd.Series | np.ndarray | Any, Any]:
    """Convert supported test inputs into true and predicted class labels.

    `Flash Explanation:` *`Use this internal helper to make classification metrics work with arrays and Keras iterators.`*
    """

    if not hasattr(model, 'model') or type(model).__name__ == 'Sequential': # ajuste seguro conforme sua lógica original
        true_model = model
    else:
        true_model = model.model

    if isinstance(x_test, DirectoryIterator):
        y_true = x_test.classes
        assert y_true is not None

        y_pred = true_model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=-1)

        return y_true, y_pred_classes

    if isinstance(x_test, NumpyArrayIterator):
        x_list, y_list = [], []
        for i in range(len(x_test)):
            batch_x, batch_y = x_test[i]
            x_list.append(batch_x)
            y_list.append(batch_y)

        x_test = np.concatenate(x_list, axis=0)
        y_test = np.concatenate(y_list, axis=0)

        y_pred = true_model.predict(x_test)
        y_pred_classes = y_pred.argmax(axis=-1)

        if y_test is None: 
            raise ValueError("``y_test`` became None at some point.")

        if y_test.ndim > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=-1)

        if y_test is None: 
            raise ValueError("``y_test`` became None at ``np.argmax(y_test, axis=-1)``.")

        return y_test, y_pred_classes
    
    if y_test is None:
        raise ValueError('``y_test`` must be provided if Test Data is not a ``BatchIterator``')

    if isinstance(x_test, pd.DataFrame):
        x_test = x_test.values

    y_pred = true_model.predict(x_test)
    y_pred_classes = y_pred.argmax(axis=-1)

    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=-1)

    if y_test is None: 
        raise ValueError("``y_test`` became None at ``np.argmax(y_test, axis=-1)``.")

    return y_test, y_pred_classes

@overload
def getAccuracy(
    model: FlashSequential | Sequential,
    x_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray | None
) -> float: """Accuracy overload for array-like test data. `Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*"""
"""Accuracy overload for array-like test data.

`Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*
"""

@overload
def getAccuracy(
    model: FlashSequential | Sequential,
    test_batches: BatchIterator
) -> float: """Accuracy overload for a batch iterator. `Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*"""
"""Accuracy overload for a batch iterator.

`Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*
"""

def getAccuracy(
    model: FlashSequential | Sequential,
    x_test: pd.DataFrame | np.ndarray | BatchIterator,
    y_test: pd.Series | np.ndarray | None = None
) -> float:
    """Return classification accuracy for a model and test data.

    `Flash Explanation:` *`Use this to measure the fraction of correctly predicted classes.`*
    """
    
    y_test, y_pred_classes = _adjustClassMetrics(model, x_test, y_test)
    
    return accuracy_score(y_test, y_pred_classes)

@overload
def getPrecision(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: """Precision overload for array-like test data. `Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*"""
"""Precision overload for array-like test data.

`Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*
"""
@overload
def getPrecision(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: """Precision overload for a batch iterator. `Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*"""
"""Precision overload for a batch iterator.

`Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*
"""
def getPrecision(model: FlashSequential | Sequential, 
                x_test: pd.DataFrame | np.ndarray | BatchIterator, 
                y_test: pd.Series | np.ndarray | None = None
                ) -> float:
    """Return macro-averaged precision for a model and test data.

    `Flash Explanation:` *`Use this to measure prediction correctness while weighting each class equally.`*
    """
    
    y_test, y_pred_classes = _adjustClassMetrics(model, x_test, y_test)
    
    return precision_score(y_test, y_pred_classes, average='macro')

@overload
def getRecall(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: """Recall overload for array-like test data. `Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*"""
"""Recall overload for array-like test data.

`Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*
"""
@overload
def getRecall(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: """Recall overload for a batch iterator. `Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*"""
"""Recall overload for a batch iterator.

`Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*
"""
def getRecall(model: FlashSequential | Sequential, 
                x_test: pd.DataFrame | np.ndarray | BatchIterator, 
                y_test: pd.Series | np.ndarray | None = None
                ) -> float:
    """Return macro-averaged recall for a model and test data.

    `Flash Explanation:` *`Use this to measure how many examples from each class are recovered.`*
    """
    
    y_test, y_pred_classes = _adjustClassMetrics(model, x_test, y_test)
    
    return recall_score(y_test, y_pred_classes, average='macro')

@overload
def getROC_AUC(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: """ROC AUC overload for array-like test data. `Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*"""
"""ROC AUC overload for array-like test data.

`Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*
"""
@overload
def getROC_AUC(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: """ROC AUC overload for a batch iterator. `Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*"""
"""ROC AUC overload for a batch iterator.

`Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*
"""
def getROC_AUC(model: FlashSequential | Sequential, 
                x_test: Union[pd.DataFrame, np.ndarray, BatchIterator], 
                y_test: Union[pd.Series, np.ndarray, None] = None
                ) -> float:
    """
    Calculate the ROC AUC score of the given model on the test data.

    `Flash Explanation:` *`Use this to measure ranking quality across classification thresholds.`*
    
    Parameters:
    model (FlashSequential | Sequential): The model to evaluate.
    x_test (Union[pd.DataFrame, np.ndarray, BatchIterator]): The test features.
    y_test (Union[pd.Series, np.ndarray, None], optional): The true labels. Defaults to None.
    
    Returns:
    float: The ROC AUC score.
    """

    y_test, y_pred_proba = _adjustClassMetrics(model, x_test, y_test)
    
    return roc_auc_score(y_test, y_pred_proba)

@overload
def getF1Score(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: """F1 overload for array-like test data. `Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*"""
"""F1 overload for array-like test data.

`Flash Explanation:` *`Use this signature when supplying explicit features and labels.`*
"""
@overload
def getF1Score(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: """F1 overload for a batch iterator. `Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*"""
"""F1 overload for a batch iterator.

`Flash Explanation:` *`Use this signature when labels are stored in the iterator.`*
"""
def getF1Score(model: FlashSequential | Sequential, 
                x_test: Union[pd.DataFrame, np.ndarray, BatchIterator], 
                y_test: Union[pd.Series, np.ndarray, None] = None
                ) -> float:
    """
    Calculate the F1 score of the given model on the test data.

    `Flash Explanation:` *`Use this to summarize macro-averaged precision and recall.`*
    
    Parameters:
    model (FlashSequential | Sequential): The model to evaluate.
    x_test (Union[pd.DataFrame, np.ndarray, BatchIterator]): The test features.
    y_test (Union[pd.Series, np.ndarray, None], optional): The true labels. Defaults to None.
    
    Returns:
    float: The F1 score.
    """

    y_test, y_pred_classes = _adjustClassMetrics(model, x_test, y_test)
    
    return f1_score(y_test, y_pred_classes, average='macro')

def getMSE(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
    """Return mean squared error for regression predictions.

    `Flash Explanation:` *`Use this to quantify the average squared prediction error.`*
    """
    if not isinstance(model, Sequential):
        true_model = model.model
    else:
        true_model = model

    y_pred = true_model.predict(x_test)
    return mean_squared_error(y_test, y_pred)

def getRMSE(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
    """Return root mean squared error for regression predictions.

    `Flash Explanation:` *`Use this to express prediction error in the target's units.`*
    """
    if not isinstance(model, Sequential):
        true_model = model.model
    else:
        true_model = model

    y_pred = true_model.predict(x_test)
    return root_mean_squared_error(y_test, y_pred)

def getMAE(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
    """Return mean absolute error for regression predictions.

    `Flash Explanation:` *`Use this to measure average absolute prediction error.`*
    """
    if not isinstance(model, Sequential):
        true_model = model.model
    else:
        true_model = model

    y_pred = true_model.predict(x_test)

    return mean_absolute_error(y_test, y_pred)