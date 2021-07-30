from abc import ABC, abstractmethod
import pandas as pd
import time, os
import math, requests

# Created as an abstract class parent from the other model needed 
class BasicModel(ABC):

    READY_STATE = 'ready'
    NOT_READY_STATE = 'not_ready'
    BUILDING_MODEL = 'building_model'

    def __init__(self) -> None:
        self._state = BasicModel.NOT_READY_STATE
        self._model = None
        self._mse = None
        self._rmspe = None
        self._test_score = None
        self._train_score = None
        self._x_scaler = None
        self._state_time = None
    
    @abstractmethod
    def train_model(self, df: pd.DataFrame):
        pass

    def predict(self, row):
        return self._model.predict(self._x_scaler.transform(row))

    def model_metrics(self) -> dict:
        metrics = {
        'state':self._state,
        'test_score':self._test_score if self._test_score is not None else 0,
        'train_score':self._train_score if self._train_score is not None else 0,
        'rmse':math.sqrt(self._mse) if self._mse is not None else 0,
        'rmspe':self._rmspe if self._rmspe is not None else 0,
        'update':time.strftime('%d-%b-%Y %H:%M:%S %Z', self._state_time)
        }
        return metrics

    def set_state_time(self):
        self._state_time = time.localtime()
    
    def send_alert(self, alert_target, data=None):
        alert_service = os.environ['alertservice_endpoint']
        requests.post(f'{alert_service}/alerts/{alert_target}', json=data)


    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model=model

    @property
    def train_score(self):
        return self._train_score
    
    @train_score.setter
    def train_score(self, train_score):
        self._train_score=train_score

    @property
    def test_score(self):
        return self._test_score
    
    @test_score.setter
    def test_score(self, test_score):
        self._test_score=test_score

    
    @property
    def mse(self):
        return self._mse
    
    @mse.setter
    def mse(self, mse):
        self._mse=mse

    @property
    def rmspe(self):
        return self._rmspe
    
    @rmspe.setter
    def rmspe(self, rmspe):
        self._rmspe=rmspe

    @property
    def x_scaler(self):
        return self._x_scaler
    
    @x_scaler.setter
    def x_scaler(self, x_scaler):
        self._x_scaler = x_scaler