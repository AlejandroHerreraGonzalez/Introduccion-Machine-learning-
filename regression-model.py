# -*- coding: utf-8 -*-

"""
Importar las diferentes dependencias. A medida que se van utilizando se van a explicar.
"""

import pandas as pd

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

"""
Indice

1) Preprocesamiento de la informacion
    Antes de realizar los modelos se debe preparar la información, ya que en muchos casos la misma se encuentra en desorden, incompleta, o con formatos inadecuados, por lo tanto uno de los aspectos mas importantes y que requiere mas tiempo en el machine learning es el preprocesamiento de la información.
    Nota : este proceso se realiza en las variables independientes.

    Tareas que se llevaran acabo en esta primer parte:
        - 1.1) Importar la base de datos
        - 1.2) Datos faltantes y partición de los datos entre variables dependientes y variables independientes
            - 1.2.1) Datos faltantes de variables categoricas
            - 1.2.2) partición de los datos entre variables dependientes y variables independientes
            - 1.2.3) Datos faltantes de variables discretas
        - 1.3) Codificación variables categóricas
            - 1.3.1) LabelEncoder
            - 1.3.1) OneHotEncoder
        - 1.4) División de los datos (entrenamiento y test)
        - 1.5) Escalado de características
2 ) Seleccion del modelo y resultados
    """
#1.1) Importar la base de datos
#   Dependencias:
#        -pandas : nos permite acceder a los datos ubicados en nuestros archivos train.csv ytest.csv, mediante pandas podemos realizar análisis de datos con un alto rendimiento.

database = pd.read_csv("/path/del/archivo/propieades.csv")


# 1.2) Datos faltantes

#     Dependencias:
#         - Pandas.
#
#     1.2.1) Datos faltantes de variables categoricas. Se reemplazan los valores faltantes con 0. Tambien se pueden eliminar o crear un modelo de machine learning aparte para predecir su valor, para este caso y con la finalidad de mantener las cosas simples se reemplazara por 0.
# Nota: Las variables categoricas son aquellas que contienen un número finito de categorías o grupos distintos, en este caso son : tipopropiedad,areaurbanas,parqueadero,centrocomercial,colegios,viasacceso,seguridad,supermercado,transportepublico,vigilancia,esquinero,estrato,nuevousado,estado,antigedad,ciudad,barrio.
# Nota: aunque la variable estrato es numerica ordinal en este caso le damos el trato de categorica. Las variabes numerohabitacines,numerobanhos tambien podrian considerarse como categoricas, en este caso no las consideramos.

database["estrato"].fillna(0,inplace = True)
database["tipopropiedad"].fillna(0,inplace = True)
database["estado"].fillna(0,inplace = True)
database["antigedad"].fillna(0,inplace = True)
database["barrio"].fillna(0,inplace = True)


#     1.2.2) Partición de los datos entre variables independientes o variables X y variables dependientes o variable Y. De esta forma se le dice al codigo cuales son las variables que se utilizaran para realizar la prediccion final.
# Nota : Cada columna del archivo propieades.csv representa una variable.
# Nota: Para nuestro caso las variables independientes o variables X son : areatotal,areaconstruida,tipopropiedad,numerohabitacines,numerobanhos,areaurbanas,parqueadero,centrocomercial,colegios,viasacceso,seguridad,supermercado,transportepublico,vigilancia,esquinero,estrato,precioadmon,nuevousado,estado,antigedad,ciudad,barrio.
# La variable dependiente o variable Y es decir la variable que queremos hallar en este caso es : precio.

x = database.iloc[:,:-1].values
y = database.iloc[:,22:23].values


#     1.2.3) Datos faltantes de variables discretas

#            Dependencias:
#            - Imputer (sklearn) : Permite completar los datos faltantes con la media de todos los datos pertenecientes a dicha variable, es decir toman los datos de toda la columna.

# Nota: Las variables discretas son aquellas que tienen un número contable de valores entre dos valores cualesquiera, en este caso dichas variables son: areatotal,areaconstruida,numerohabitacines,numerobanhos, precio
# Nota: Si la variable dependiente no cuenta con un valor se podria o eliminar dicha serie de datos o asignarles un valor correspondiente al valor de la media de todos los datos en el caso de que sea una variable numerica (esta ultima opcion puede afectar en gran medida los resultados del modelo final por lo tanto es recomendable eliminarlos), en nuestro caso la variable dependiente no tiene datos faltantes.

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)

imputer1 = imputer.fit(x[:,0:2])
x[:,0:2] = imputer1.transform(x[:,0:2])

imputer2 = imputer.fit(x[:,3:5])
x[:,3:5] = imputer2.transform(x[:,3:5])

imputer3 = imputer.fit(x[:,16:17])
x[:,16:17] = imputer3.transform(x[:,16:17])


# 1.3) Codificación variables categóricas

#       - 1.3.1) LabelEncoder
#                Dependencias:
#                - LabelEncoder (sklearn) : Permite asignar un valor numerico ordinal a los diferentes datos dentro de las variables categoricas, de esta forma el modelo puede utilizar dicha informacion sin arrojar un error.
 # Nota: Se debe convertir las variables categoricas en valores numericos con la finalidad de que se puedan realizar los calculos ya que no se pueden multiplicar numeros por letras.
 # Nota : Este metodo puede causar sesgos en ciertos algoritmos dado a la forma en que son creados los valores numericos, por lo tanto se recomiendo utilizar este metodo en conjunto con OneHotEncoder.


labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()
labelencoder_x_4 = LabelEncoder()
labelencoder_x_5 = LabelEncoder()
labelencoder_x_6 = LabelEncoder()
labelencoder_x_7 = LabelEncoder()
labelencoder_x_8 = LabelEncoder()
labelencoder_x_9 = LabelEncoder()
labelencoder_x_10 = LabelEncoder()
labelencoder_x_11 = LabelEncoder()
labelencoder_x_12 = LabelEncoder()
labelencoder_x_13 = LabelEncoder()
labelencoder_x_14 = LabelEncoder()
labelencoder_x_15 = LabelEncoder()
labelencoder_x_16 = LabelEncoder()
labelencoder_x_17 = LabelEncoder()
labelencoder_x_18 = LabelEncoder()


x[:,2] = labelencoder_x_1.fit_transform(x[:,2])
x[:,5] = labelencoder_x_2.fit_transform(x[:,5])
x[:,6] = labelencoder_x_3.fit_transform(x[:,6])
x[:,7] = labelencoder_x_4.fit_transform(x[:,7])
x[:,8] = labelencoder_x_5.fit_transform(x[:,8])
x[:,9] = labelencoder_x_6.fit_transform(x[:,9])
x[:,10] = labelencoder_x_7.fit_transform(x[:,10])
x[:,11] = labelencoder_x_8.fit_transform(x[:,11])
x[:,12] = labelencoder_x_9.fit_transform(x[:,12])
x[:,13] = labelencoder_x_10.fit_transform(x[:,13])
x[:,14] = labelencoder_x_11.fit_transform(x[:,14])
x[:,15] = labelencoder_x_12.fit_transform(x[:,15])
x[:,17] = labelencoder_x_14.fit_transform(x[:,17])
x[:,18] = labelencoder_x_15.fit_transform(x[:,18])
x[:,19] = labelencoder_x_16.fit_transform(x[:,19])
x[:,20] = labelencoder_x_17.fit_transform(x[:,20])
x[:,21] = labelencoder_x_18.fit_transform(x[:,21])

# Con esto nos aseguramos que todos los datos de las variables categoricas se encuentran en valores numericos ordinales

#       - 1.3.1) OneHotEncoder
#                Dependencias:
#                - OneHotEncoder (sklearn) : Convierte los valores ordinales obtenidos en el paso anterior en valores binarios, los cuales son utiles a la hora de evitar sesgos en los diferentes algoritmos.
# Nota: Dado que las variables areaurbanas,parqueadero,centrocomercial,colegios,viasacceso,seguridad,supermercado,transportepublico,vigilancia,esquinero,nuevousado son binarias (SI,NO) no es necesario aplicar OneHotEncoder en ellas.
# Nota: OneHotEncoder creara una nueva columna por cada dato ordinal en las variables por lo tanto pasara de 22 columnas correspondiente a 22 variables independientes a 46 columnas.

# print len(x[0])

onehotencoder = OneHotEncoder(categorical_features=[2,17,18,19,20,21])
x = onehotencoder.fit_transform(x).toarray()

# print len(x[0])

# 1.4) División de los datos (entrenamiento y test)
#      Dependencias:
#      - train_test_split (sklearn):  Permite dividir nuestra base de datos de forma aleatoria en una parte de entrenamiento y otra de prueba, con esto nos aseguramos que el modelo pueda entrenarse y luego ponerse a prueba en datos completamente nuevos.

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state =2)

print "valores reales "
print y_test[0:20]


# 1.5) Escalado de características
#      Dependencias:
#      - StandardScaler (sklearn): Evita sesgos a la hora de realizar los diferentes algoritmos colocando todos los valores numericos un una misma escala, adicional mejora el rendimiento en cuanto al calculo de las operaciones.
# Nota: Se debe realizar el proceso tanto para las variables independientes como para las dependientes en la parte de entrenamiento y test.

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)


# 2 ) Seleccion del modelo y resultados
#     Dependencias:
#     - RandomForestRegressor (sklearn): Permite utilizar un modelo de regresion, existe una gran cantidad de modelos, por simpleza utilizaremos este.
#     - StandardScaler (sklearn)
#     - r2_score (sklearn) : Permite validar el rendimiento del modelo, entre mas cerca este de 1 mas presiso sera el modelo.

#     Nota: Cada modelo tiene sus propios parametros los cuales se deben ajustar dependiendo de los resultados que arroje. No hay otra forma de saber cuales son los parametros optimos ademas del ensayo y el error.

regresor = RandomForestRegressor(n_estimators=700,criterion="mse",n_jobs=-1,random_state=2,max_features=9,max_leaf_nodes=10000,bootstrap=True,min_impurity_split=0.000001,min_samples_split=20)
regresor.fit(x_train,y_train.ravel())

y_pred = regresor.predict(x_test)
y_pred_train = regresor.predict(x_train)


print "Prediccion del modelo"
print sc_y.inverse_transform(y_pred)[0:20]


# Entre mas cerca esten estos valores a 1 mejor ya que indican el % de asertividad de nuestro modelo.
# los valores r2_score_train y los valores r2_score_test deberian estar muy cerca el uno del otro ya que de lo contrario se presentaria overfitting o underfitting.
print "r2_score_train "
print r2_score(y_train,y_pred_train)
print "r2_score_test "
print r2_score(y_test,y_pred)

# Resultados
# r2_score_train = 0.838795075521
# r2_score_test = 0.728394005102
