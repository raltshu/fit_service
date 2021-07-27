import json
from flask_classful import FlaskView, route
from flask import request,Response

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
        pass

    @route('/status', methods=['GET'])
    def get_status(self):
        if DiamondsModel._main_model is None:
            result = {'state':BasicModel.NOT_READY_STATE}
        else:
            result = DiamondsModel._main_model.get_status()
        
        return json.dumps(result)
