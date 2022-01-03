from abc import abstractmethod
from resultado import Resultado
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from typing import List,Union
class MetodoAprendizadoDeMaquina:

    @abstractmethod
    def eval(self,df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str) -> Resultado:
        raise NotImplementedError

class ScikitLearnAprendizadoDeMaquina(MetodoAprendizadoDeMaquina):
    def __init__(self,ml_method:Union[ClassifierMixin,RegressorMixin]):
        self.ml_method = ml_method

    def eval(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, seed:int=1) -> Resultado:
        x_treino = df_treino.drop(labels=col_classe, axis=1)
        y_treino = df_treino[col_classe]

        model = self.ml_method.fit(x_treino, y_treino)
        
        x_to_predict = df_data_to_predict.drop(labels=col_classe, axis=1)
        y_to_predict = df_data_to_predict[col_classe]

        y_predictions = self.ml_method.predict(x_to_predict)
        return Resultado(y_to_predict,y_predictions)
