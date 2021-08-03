import pandas as pd
import numpy as np
import seaborn as sb
import os.path
import os
import pickle
import bz2
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from models.basic_model import BasicModel
from models.cut_model import CutModel
from models.full_data_model import FullDataModel


class DiamondsModel(BasicModel):
  # This would be a static reference to the DiamondsModel for easy access
  MODEL_FILE_PATH = os.getcwd()+"/files/diamonds_model_file"  
  _main_model = None

  def __init__(self) -> None:
      super().__init__()
      self._cut_models = dict()

  def get_status(self) -> dict:
    status = super().model_metrics()
    return status

  def train_model(self, df: pd.DataFrame, outsource_data=False) -> None :
      if outsource_data:
        self.send_alert('train_outsource_model_start',{'rows_number':df.shape[0]})
      else:
        self.send_alert('train_model_start')

      self._state = BasicModel.BUILDING_MODEL
      #Cleanup data
      df = df.query('x >0 and y>0 and z>0').copy()
      #Remove outliers
      df = df.query('y<=10').copy()
      df = df.query('z<=6 and z>=2').copy()

      #Create a new 'volume' column which replaces x,y,z
      df = DiamondsModel.add_volume_column(df.copy())
      
      #Replace categorial features with numeric values
      df = DiamondsModel.replace_categorial_values(df.copy())
      
      # Create a regression model for each cut value
      for cut_group in df['cut'].unique():
        cut_model = CutModel()
        cut_model.train_model(df, cut_group)
        self._cut_models[cut_group] = cut_model

      #Train a model cross all data
      self._full_data_model = FullDataModel()
      self._full_data_model.train_model(df)

      self._state=BasicModel.READY_STATE
      self.set_state_time()
      if not outsource_data:
        self.store_to_file()
        self.send_alert('train_model_finish', self.model_metrics())
      else:
        self.send_alert('train_outsource_model_finish', self.model_metrics())
  
  def predict(self, row: pd.Series) -> float:

    if row.get('price') is not None:
      target=row['price']
      row = row.drop(['price'])

    if not DiamondsModel.validate_row(row):
      pass

    if DiamondsModel.outliers_found(row):
      pass

    # Transform to Dataframe for the below operarions
    row = row.to_frame().T
    row = DiamondsModel.replace_categorial_values(row)
    row = DiamondsModel.add_volume_column(row)
    # Transform back to Series
    row = row.iloc[0].copy()

    cut_value = row['cut']
    row = row.values.reshape(1,-1)
    
    y2 = self._full_data_model.predict(row)
    
    cut_model_name = f'{cut_value}_model'
    y1 = self._cut_models[cut_value].predict(row)
    
    y = 0.8*y1[0] + 0.2*y2[0]
    
    return y

  def calc_score(self, df: pd.DataFrame, outsource_data=False) -> None:
    if outsource_data:
      self.send_alert('calc_outsource_score_begin',{'rows_number':df.shape[0]})
    else:
      self.send_alert('calc_score_begin')

    #Cleanup data
    df = df.query('x >0 and y>0 and z>0').copy()
    #Remove outliers
    df = df.query('y<=10').copy()
    df = df.query('z<=6 and z>=2').copy()
    
    if not outsource_data:
      X_train, X_test  = train_test_split(df,test_size=0.1, random_state=50)
    else:
      X_test = df

    predictions = X_test.apply(self.predict,axis=1,result_type='expand')
    predictions = predictions.to_frame()
    predictions.columns = ['y']

    self._test_score = r2_score(X_test['price'],predictions['y'])
    self._mse = mean_squared_error(X_test[['price']],predictions[['y']])
    
    y_test = X_test['price']
    y_pred = predictions['y']
    self._rmspe = (np.sqrt(np.mean(np.square((y_test - y_pred) / y_test)))) * 100
    
    self._state=BasicModel.READY_STATE
    self.set_state_time()
    if not outsource_data:
      self.store_to_file()
      self.send_alert('calc_score_complete', self.model_metrics())
    else:
      self.send_alert('calc_outsource_score_complete', self.model_metrics())

  @staticmethod
  def detect_outliers(df: pd.DataFrame):
    diamonds_org = sb.load_dataset('diamonds')
    cut_list = diamonds_org['cut'].unique()
    color_list = diamonds_org['color'].unique()
    clarity_list = diamonds_org['clarity'].unique()
    outliers_df = df.query('x<=0 or y<=0 or y>10 or z<2 or z>6 or\
      cut not in @cut_list or \
      color not in @color_list or\
      clarity not in @clarity_list').copy()
    non_outliers_df = df.query('x>0 and y>0 and y<=10 and z>=2 and z<=6 and\
      cut in @cut_list and \
      color in @color_list and\
      clarity in @clarity_list').copy()
    return non_outliers_df, outliers_df



  @staticmethod
  def validate_row(row: pd.Series) -> bool:
    #Check if x,y or z have zero values
    validation = row.x*row.y*row.z > 0
    return validation

  @staticmethod
  def outliers_found(row: pd.Series) -> bool:
    outliers = row.y > 10
    outliers = row.z > 6 or row.y < 2
    return outliers

  @staticmethod
  def replace_categorial_values(row: pd.DataFrame) -> pd.DataFrame:
    row_temp = row.copy()
    row_temp['cut'].replace({'Ideal':1,'Premium':2,'Very Good':3,'Good':4,'Fair':5}, inplace=True)
    row_temp['clarity'].replace({'IF':1,'VVS1':2,'VVS2':3,'VS1':4,'VS2':5,'SI1':6,'SI2':7,'I1':8}, inplace=True)
    row_temp['color'].replace({'D':1,'E':2,'F':3,'G':4,'H':5,'I':6,'J':7},inplace=True)

    return row_temp


  @staticmethod
  def add_volume_column(row: pd.DataFrame) -> pd.DataFrame:
    row_temp = row.copy()
    row_temp['volume'] = row.x*row.y*row.z
    row_temp.drop(['x','y','z'], axis=1, inplace=True)

    return row_temp

  @staticmethod
  def load_from_file() -> None:
    if os.path.exists(DiamondsModel.MODEL_FILE_PATH):
      with bz2.BZ2File(DiamondsModel.MODEL_FILE_PATH,'r') as file:
          DiamondsModel._main_model = pickle.load(file)
  
  def store_to_file(self) -> None:
    DiamondsModel._main_model = self
    with bz2.BZ2File(DiamondsModel.MODEL_FILE_PATH,'w') as file:
        pickle.dump(self, file)
  
  @staticmethod
  def is_model_ready() -> bool:
    return DiamondsModel._main_model is not None





