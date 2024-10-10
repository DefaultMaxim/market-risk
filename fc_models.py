import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor


class CatboostModel:
    
    def __init__(
        self,
        n_splits: int=8,
        iterations: int=1_000,
        depth: int=7,
        learning_rate: float=0.05,
        verbose: bool = False,
    ):
        
        self.n_splits = n_splits
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.verbose = verbose
    
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> tuple:
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        y_pred_history = []
        y_test_history = []
        
        for fold,(train_index, test_index) in enumerate(tscv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, self.y_test = y[train_index], y[test_index]

            self.model = CatBoostRegressor(
                iterations=self.iterations, 
                depth=self.depth, 
                learning_rate=self.learning_rate
                )
            
            self.model.fit(X_train, y_train, verbose=self.verbose)
            
            # self.y_pred = self.model.predict(X_test)
        
            # y_pred_history.append(self.model.predict(X_test))
            # y_test_history.append(self.y_test)
            
            # self.y_pred_history = y_pred_history
            # self.y_test_history = y_test_history

    def predict(
        self,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        
        return self.model.predict(X_test)