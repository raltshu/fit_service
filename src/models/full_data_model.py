from models.basic_model import BasicModel
import pandas as pd
import numpy as np
import sklearn.preprocessing as sp
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class FullDataModel(BasicModel):
    def __init__(self) -> None:
        super().__init__()

    def train_model(self, df: pd.DataFrame):
        X = df.drop(['price'],axis=1)
        y = df['price']
        x_scaler = sp.StandardScaler() 

        m = RandomForestRegressor(n_estimators=200,random_state=40,min_samples_leaf=5) 

        x_scaler = sp.StandardScaler() 

        X = x_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        
        m.fit(X_train,y_train)
        y_pred = m.predict(X_test)

        self._test_score = m.score(X_test, y_test)
        self._train_score = m.score(X_train, y_train)
        self._mse = mean_squared_error(y_test, y_pred)
        self._rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred) / y_test)))) * 100
        self._model = m
        self._x_scaler = x_scaler
        self.set_state_time()
        self._state = BasicModel.READY_STATE