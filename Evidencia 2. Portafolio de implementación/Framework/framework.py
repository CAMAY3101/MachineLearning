#importing libraries
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import seaborn as sb
import plotly.express as px
import sklearn as sk
import matplotlib.pyplot as plt

# import the data
columns = ["class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",  "Flavanoids","Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv('../datasets/wine.data', names=columns)
print(df.head())

#exploring the data

#data types
print(df.info())

#statistical summary
print(df.describe())

#inconsistencies
print(df.isnull().sum())
print(df.isna().sum())

#Correlacion de variables
corr = df.corr()  # Coeficiente de Pearson
sb.heatmap(corr)

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sb.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sb.heatmap(corr, mask=mask, vmax=1, square=True)

# Unicamente se seleccionan las columnas que contienen una correlacion mayor a 0.75
selections = ["Total phenols", "Flavanoids", "OD280/OD315 of diluted wines"]
dfNew = df[selections]
g = sb.PairGrid(dfNew)
g.map(plt.scatter)


#modeling

x = df[["Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",  "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]].values
y = df[['Alcohol']].values

X_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

modelo_regresion = LinearRegression()  # modelo de regresión

# aprendizaje automático con base en nuestros datos
modelo_regresion.fit(X_train, y_train)


x_columns = ["Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",  "Flavanoids",
             "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
coeff_df = pd.DataFrame(modelo_regresion.coef_[0],
                        x_columns, columns=['Coeficientes'])
coeff_df  # despliega los coefientes y sus valores; por cada unidad del coeficente, su impacto en las calorías será igual a su valor


# probamos nuestro modelo con los valores de prueba
y_pred = modelo_regresion.predict(x_test)


# creamos un dataframe con los valores actuales y los de predicción
validacion = pd.DataFrame({'Alcohol (Actual)': y_test.reshape(1, 36)[0],
                           'Alcohol (Predicción)': y_pred.reshape(1, 36)[0],
                           'Diferencia': y_test.reshape(1, 36)[0] - y_pred.reshape(1, 36)[0]})

muestra_validacion = validacion.head(25)  # elegimos una muestra con 25 valores

validacion["Diferencia"].describe()


r2_score(y_test, y_pred)  # ingresamos nuestros valores reales y calculados

# creamos un gráfico de barras con el dataframe que contiene nuestros datos actuales y de predicción
muestra_validacion.plot.bar(rot=0)

# indicamos el título del gráfico
plt.title("Comparación de Alcohol actual y de predicción")

# indicamos la etiqueta del eje de las x
plt.xlabel("Muestra")

# indicamos la etiqueta del eje de las y
plt.ylabel("Cantidad de Alcohol")

plt.show()  # desplegamos el gráfico

print('Puntaje del r2 ', r2_score(y_test, y_pred))
print('Error promedio ', mean_squared_error(y_test, y_pred))
print('Error de la raiz cuadrada del promedio of is',
      np.sqrt(mean_squared_error(y_test, y_pred)))


#Se corren al menos cinco predicciones para validar la salida del modelo

data_xtest = pd.DataFrame(x_test, columns=x_columns)
predictions = pd.concat([data_xtest, validacion], axis=1)
predictions.head(5)
