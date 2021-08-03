import threading
import os, time, requests, json
import pandas as pd

from models.diamonds_model import DiamondsModel

dataservice = os.environ['dataservice_endpoint']

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
        


class ModelsManager(metaclass=Singleton):

    __q = []

    def add_job(self, job):
        if job not in self.__q:
            self.__q.append(job)
    
    def __run_job(self, job):
        inner_thread = threading.Thread(target=job, daemon=True)
        inner_thread.start()
        inner_thread.join()
    
    @staticmethod
    def train_model():
        diamonds_model = DiamondsModel()
        response = requests.get(url=f"{dataservice}/data/table_view/diamonds_org")
        data = json.loads(response.text)
        df = pd.read_json(data['df'])        #Drop the Index column that came from the DB
        df = df.iloc[: , 1:]
        diamonds_model.train_model(df)

    @staticmethod
    def train_outsource_model():
        diamonds_model = DiamondsModel()
        response = requests.get(url=f"{dataservice}/data/table_view/outsource_diamonds?rand=2000")
        data = json.loads(response.text)
        df = pd.read_json(data['df'])

        non_outliers_df, outliers_df = diamonds_model.detect_outliers(df)
        requests.post(url=f"{dataservice}/data/store_outliers", json=outliers_df.to_json())
        non_outliers_df.drop(['index','record_time'], axis=1, inplace=True)
        diamonds_model.train_model(non_outliers_df, outsource_data=True)
        ModelsManager.calc_outsource_score(model=diamonds_model)

    @staticmethod
    def calc_score():
        diamonds_model = DiamondsModel._main_model
        if diamonds_model is not None:
            response = requests.get(url=f"{dataservice}/data/table_view/diamonds_org")
            data = json.loads(response.text)
            df = pd.read_json(data['df'])
            #Drop the index column that came from the DB
            df = df.iloc[: , 1:]
            diamonds_model.calc_score(df)

    @staticmethod
    def calc_outsource_score(model=None):
        diamonds_model = model if model is not None else DiamondsModel._main_model
        if diamonds_model is not None:
            response = requests.get(url=f"{dataservice}/data/table_view/outsource_diamonds?rand=1000")
            data = json.loads(response.text)
            df = pd.read_json(data['df'])

            non_outliers_df, outliers_df = diamonds_model.detect_outliers(df)
            requests.post(url=f"{dataservice}/data/store_outliers", json=outliers_df.to_json())
            non_outliers_df.drop(['index','record_time'], axis=1, inplace=True)
            diamonds_model.calc_score(non_outliers_df, outsource_data=True)    
        
    def check_queue(self):
        while True:
            time.sleep(5)
            if self.__q:
                self.__run_job(self.__q[0])
                self.__q.pop(0)

    
t = threading.Thread(target=ModelsManager().check_queue, daemon=True)
t.start()

    