import threading
import os, time, requests, logging
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
        logging.error("In the model")
        diamonds_model = DiamondsModel()
        response = requests.get(url=f"{dataservice}/data/table_view/diamonds_org")
        df = pd.read_json(response.text)
        #Drop the Index column that came from the DB
        df = df.iloc[: , 1:]
        diamonds_model.train_model(df)

    @staticmethod
    def calc_score():
        diamonds_model = DiamondsModel._main_model
        if diamonds_model is not None:
            response = requests.get(url=f"{dataservice}/data/table_view/diamonds_org")
            df = pd.read_json(response.text)
            #Drop the index column that came from the DB
            df = df.iloc[: , 1:]
            diamonds_model.calc_score(df)

    def check_queue(self):
        while True:
            time.sleep(5)
            if self.__q:
                self.__run_job(self.__q[0])
                self.__q.pop(0)

    
t = threading.Thread(target=ModelsManager().check_queue, daemon=True)
t.start()

    