from flashkeras.utils.otherimports import *
from flashkeras.utils.kerasimports import *
from flashkeras.utils.typehints import *
from flashkeras.models import FlashSequential 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, f1_score, roc_curve # type: ignore

class FlashEvaluating:

    @staticmethod
    def _adjustClassMetrics(model: FlashSequential | Sequential, 
                            x_test: pd.DataFrame | np.ndarray | BatchIterator, 
                            y_test: pd.Series | np.ndarray | None = None
                            ) -> tuple[pd.Series | np.ndarray, Any]:

        if not isinstance(model, Sequential):
            true_model = model.model
        else:
            true_model = model

        if isinstance(x_test, (DirectoryIterator, NumpyArrayIterator)):
            x_list = [x for x, _ in x_test]
            y_list = [y for _, y in x_test]
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

    @staticmethod
    @overload
    def getAccuracy(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray | None) -> float: ...
    @staticmethod
    @overload
    def getAccuracy(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: ...
    @staticmethod
    def getAccuracy(model: FlashSequential | Sequential, 
                    x_test: pd.DataFrame | np.ndarray | BatchIterator, 
                    y_test: pd.Series | np.ndarray | None = None
                    ) -> float:
        
        y_test, y_pred_classes = FlashEvaluating._adjustClassMetrics(model, x_test, y_test)
        
        return accuracy_score(y_test, y_pred_classes)

    @staticmethod
    @overload
    def getPrecision(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: ...
    @staticmethod
    @overload
    def getPrecision(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: ...
    @staticmethod
    def getPrecision(model: FlashSequential | Sequential, 
                    x_test: pd.DataFrame | np.ndarray | BatchIterator, 
                    y_test: pd.Series | np.ndarray | None = None
                    ) -> float:
        
        y_test, y_pred_classes = FlashEvaluating._adjustClassMetrics(model, x_test, y_test)
        
        return precision_score(y_test, y_pred_classes, average='macro')
    
    @staticmethod
    @overload
    def getRecall(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: ...
    @staticmethod
    @overload
    def getRecall(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: ...
    @staticmethod
    def getRecall(model: FlashSequential | Sequential, 
                    x_test: pd.DataFrame | np.ndarray | BatchIterator, 
                    y_test: pd.Series | np.ndarray | None = None
                    ) -> float:
        
        y_test, y_pred_classes = FlashEvaluating._adjustClassMetrics(model, x_test, y_test)
        
        return recall_score(y_test, y_pred_classes, average='macro')

    @staticmethod
    @overload
    def getROC_AUC(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: ...
    @staticmethod
    @overload
    def getROC_AUC(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: ...
    @staticmethod
    def getROC_AUC(model: FlashSequential | Sequential, 
                   x_test: Union[pd.DataFrame, np.ndarray, BatchIterator], 
                   y_test: Union[pd.Series, np.ndarray, None] = None
                   ) -> float:
        """
        Calculate the ROC AUC score of the given model on the test data.
        
        Parameters:
        model (FlashSequential | Sequential): The model to evaluate.
        x_test (Union[pd.DataFrame, np.ndarray, BatchIterator]): The test features.
        y_test (Union[pd.Series, np.ndarray, None], optional): The true labels. Defaults to None.
        
        Returns:
        float: The ROC AUC score.
        """

        y_test, y_pred_proba = FlashEvaluating._adjustClassMetrics(model, x_test, y_test)
        
        return roc_auc_score(y_test, y_pred_proba)

    @staticmethod
    @overload
    def getF1Score(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray) -> float: ...
    @staticmethod
    @overload
    def getF1Score(model: FlashSequential | Sequential, x_test: BatchIterator) -> float: ...
    @staticmethod
    def getF1Score(model: FlashSequential | Sequential, 
                   x_test: Union[pd.DataFrame, np.ndarray, BatchIterator], 
                   y_test: Union[pd.Series, np.ndarray, None] = None
                   ) -> float:
        """
        Calculate the F1 score of the given model on the test data.
        
        Parameters:
        model (FlashSequential | Sequential): The model to evaluate.
        x_test (Union[pd.DataFrame, np.ndarray, BatchIterator]): The test features.
        y_test (Union[pd.Series, np.ndarray, None], optional): The true labels. Defaults to None.
        
        Returns:
        float: The F1 score.
        """

        y_test, y_pred_classes = FlashEvaluating._adjustClassMetrics(model, x_test, y_test)
        
        return f1_score(y_test, y_pred_classes, average='macro')

    @staticmethod
    def getMSE(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
        if not isinstance(model, Sequential):
            true_model = model.model
        else:
            true_model = model

        y_pred = true_model.predict(x_test)
        return mean_squared_error(y_test, y_pred)
    
    @staticmethod
    def getRMSE(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
        if not isinstance(model, Sequential):
            true_model = model.model
        else:
            true_model = model

        y_pred = true_model.predict(x_test)
        return root_mean_squared_error(y_test, y_pred)

    @staticmethod
    def getMAE(model: FlashSequential | Sequential, x_test: pd.DataFrame | np.ndarray, y_test: pd.Series | np.ndarray):
        if not isinstance(model, Sequential):
            true_model = model.model
        else:
            true_model = model

        y_pred = true_model.predict(x_test)

        return mean_absolute_error(y_test, y_pred)