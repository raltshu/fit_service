from flask import Flask
from handlers.fit_proxy import ModelView
from models.diamonds_model import DiamondsModel


app = Flask(__name__)

ModelView.register(app, route_base='/fit')
# If a model is already stored on disk, try to read it
DiamondsModel.load_from_file()


# if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5001)


