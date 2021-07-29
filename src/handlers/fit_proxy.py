import json, logging
from flask_classful import FlaskView, route
from flask import request,Response
import pandas as pd
import numpy as np

from models.basic_model import BasicModel
from models.diamonds_model import DiamondsModel
from models.models_manager import ModelsManager


class ModelView(FlaskView):

    def index(self):
        return ('Hello')

    @route('/train_model', methods=['GET'])
    def train_model(self):        
        ModelsManager().add_job(ModelsManager.train_model)
        return Response(
            "Building...",
            mimetype='application/json',
            status=200
        )

    @route('/calc_score', methods=['GET'])
    def calc_score(self):
        ModelsManager().add_job(ModelsManager.calc_score)
        return Response(
            "Calculating...",
            mimetype='application/json',
            status=200
        )

    @route('/predict',methods=['POST'])
    def predict(self):
        row = request.get_json()
        # Expecting a diamonds row representation in the following order:
        # carat, depth, table, cut, color, clarity, x, y, z
        df = pd.DataFrame([row])
        df = df.astype(dtype={'carat':float, 'depth':float, 'table':float, 
                               'cut':str, 'color':str,'clarity':str,
                               'x':float, 'y':float, 'z':float})
        result = DiamondsModel._main_model.predict(df.iloc[0])
        return str(result)

    @route('/status', methods=['GET'])
    def get_status(self):
        if DiamondsModel._main_model is None:
            result = {'state':BasicModel.NOT_READY_STATE}
        else:
            result = DiamondsModel._main_model.get_status()
        
        return json.dumps(result)
