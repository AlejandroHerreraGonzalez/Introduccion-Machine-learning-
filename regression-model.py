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

    Tareas que se llevaran acabo en esta primer parte:
        - 1.1) Importar la base de datos
        - 1.2) Datos faltantes y partición de los datos entre variables dependientes y variables independientes
            - 1.2.1) Datos faltantes de variables categoricas
            - 1.2.2) partición de los datos entre variables dependientes y variables independientes
            - 1.2.3) Datos faltantes de variables numericas
        - 1.3) Variables categóricas
        - 1.4) División de los datos (entrenamiento y test)
        - 1.5) Escalado de características

    """
#1.1) Importar la base de datos
#   Dependencias:
#        -pandas : nos permite acceder a los datos ubicados en nuestros archivos train.csv ytest.csv, mediante pandas podemos realizar análisis de datos con un alto rendimiento.

database = pd.read_csv("/home/alejandro/Escritorio/documentos/datos-github-train.csv")


# 1.2) Datos faltantes

#     Dependencias:
#         - Pandas.
#         - Imputer (sklearn) : Permite completar los valores faltantes con la media de todos los valores pertenecientes a dicha variable


#     1.2.1) Datos faltantes de variables categoricas. Se reemplazan los valores faltantes con 0. Tambien se pueden eliminar o crear un modelo de machine learning aparte para predecir su valor, para este caso y con la finalidad de mantener las cosas simples se reemplazara por 0.
# Nota: aunque la variable estrato es numerica en este caso le damos el trato de categorica.

database["estrato"].fillna(0,inplace = True)
database["tipopropiedad"].fillna(0,inplace = True)
database["estado"].fillna(0,inplace = True)
database["antigedad"].fillna(0,inplace = True)
database["barrio"].fillna(0,inplace = True)


#     1.2.2) partición de los datos entre variables dependientes y variables independientes. De esta forma se le dice al codigo cuales son las variables que se utilizaran para realizar la prediccion final.
# Nota: Para nuestro caso las variables independientes son : areatotal,areaconstruida,tipopropiedad,numerohabitacines,numerobanhos,areaurbanas,parqueadero,centrocomercial,colegios,viasacceso,seguridad,supermercado,transportepublico,vigilancia,esquinero,estrato,precioadmon,nuevousado,estado,antigedad,ciudad,barrio.
# Y la variable dependiente es decir la variable que queremos hallar en este caso es : precio


#     1.2.2) Datos faltantes de variables numericas





# 1.1.2) Datos faltantes de variables numericas

# imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
# # print x[1:2]
# imputer = imputer.fit_transform(x[:,0:2])
# # print x[:,0:2][0:10]
# x[:,0:2] = imputer.transform(x[:,0:2])
# # print x[:,0:2][0:10]
#
# #
# imputer3 = imputer.fit(x[:,3:5])
# # print x[:,15:17][0:20]
# # print "kfkkffkfkfkfkfkfk"
# x[:,3:5] = imputer3.transform(x[:,3:5])
# # print x[:,15:17][0:20]
# #
# #
# # imputer4 = imputer.fit(x[:,15:16])
# # # print x[:,15:17][0:20]
# # # print "kfkkffkfkfkfkfkfk"
# # x[:,15:16] = imputer4.transform(x[:,15:16])
# imputer4 = imputer.fit(x[:,6:7])
# # # print x[:,15:17][0:20]
# # # print "kfkkffkfkfkfkfkfk"
# x[:,6:7] = imputer4.transform(x[:,6:7])
# # print x


# print database
