# -*- coding: utf-8 -*-

"""
Importar las diferentes dependencias. A medida que se van utilizando se van a explicar.
"""
import  numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

"""
1) Preprocesamiento de la informacion 
    Antes de realizar los modelos se debe preparar la información, ya que en muchos casos la misma se encuentra en desorden, incompleta, o con formatos inadecuados, por lo tanto uno de los aspectos mas importantes y que requiere mas tiempo en el machine learning es el preprocesamiento de la información. 
"""

database = pd.read_csv("/home/alejandro/Escritorio/documentos/datos-github-train.csv")

