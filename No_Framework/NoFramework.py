import matplotlib.pyplot as plt  # importamos la librería que nos permitirá graficar
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats import f
from scipy import stats

columns = ["class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",  "Flavanoids",
           "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv('datasets/wine.data', names=columns)
df.head()

#class for a regression model for a pandas dataframe for more than two independent variables and one dependent variable


class RegressionModel:
    def __init__(self, df, dependent_variable, independent_variables):
        self.df = df
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables
        self.coefficients = self.get_coefficients()
        self.intercept = self.get_intercept()
        self.predicted = self.get_predicted()
        self.residuals = self.get_residuals()
        self.r_squared = self.get_r_squared()
        self.f_statistic = self.get_f_statistic()
        self.f_p_value = self.get_f_p_value()
        self.t_statistics = self.get_t_statistics()
        self.t_p_values = self.get_t_p_values()
        self.summary = self.get_summary()

    def get_coefficients(self):
        return np.linalg.inv(self.df[self.independent_variables].T.dot(self.df[self.independent_variables])).dot(self.df[self.independent_variables].T).dot(self.df[self.dependent_variable])

    def get_intercept(self):
        return np.mean(self.df[self.dependent_variable]) - np.mean(self.df[self.independent_variables].dot(self.coefficients))

    def get_predicted(self):
        return self.df[self.independent_variables].dot(self.coefficients) + self.intercept

    def get_residuals(self):
        return self.df[self.dependent_variable] - self.predicted

    def get_r_squared(self):
        return 1 - (np.sum(self.residuals ** 2) / np.sum((self.df[self.dependent_variable] - np.mean(self.df[self.dependent_variable])) ** 2))

    def get_f_statistic(self):
        return (self.r_squared / (len(self.independent_variables) - 1)) / ((1 - self.r_squared) / (len(self.df) - len(self.independent_variables)))

    def get_f_p_value(self):
        return 1 - stats.f.cdf(self.f_statistic, len(self.independent_variables) - 1, len(self.df) - len(self.independent_variables))

    def get_t_statistics(self):
        return self.coefficients / np.sqrt(np.diag(np.linalg.inv(self.df[self.independent_variables].T.dot(self.df[self.independent_variables]))))

    def get_t_p_values(self):
        return 2 * (1 - stats.t.cdf(np.abs(self.t_statistics), len(self.df) - len(self.independent_variables)))

    def get_summary(self):
        summary = pd.DataFrame
        return summary

#class for a regression model for a pandas dataframe for more than two independent variables and one dependent variable

class RegressionModel:
    def __init__(self, df, dependent_variable, independent_variables):
        self.df = df
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables
        self.coefficients = self.get_coefficients()
        self.intercept = self.get_intercept()
        self.predicted = self.get_predicted()
        self.residuals = self.get_residuals()
        self.r_squared = self.get_r_squared()
        self.f_statistic = self.get_f_statistic()
        self.f_p_value = self.get_f_p_value()
        self.t_statistics = self.get_t_statistics()
        self.t_p_values = self.get_t_p_values()
        self.summary = self.get_summary()
        
    def get_coefficients(self):
        return np.linalg.inv(self.df[self.independent_variables].T.dot(self.df[self.independent_variables])).dot(self.df[self.independent_variables].T).dot(self.df[self.dependent_variable])
    def get_intercept(self):
        return np.mean(self.df[self.dependent_variable]) - np.mean(self.df[self.independent_variables].dot(self.coefficients))
    
    def get_predicted(self):
        return self.df[self.independent_variables].dot(self.coefficients) + self.intercept
    
    def get_residuals(self):
        return self.df[self.dependent_variable] - self.predicted
    
    def get_r_squared(self):
        return 1 - (np.sum(self.residuals ** 2) / np.sum((self.df[self.dependent_variable] - np.mean(self.df[self.dependent_variable])) ** 2))
    
    def get_f_statistic(self):
        return (self.r_squared / (len(self.independent_variables) - 1)) / ((1 - self.r_squared) / (len(self.df) - len(self.independent_variables)))
    
    def get_f_p_value(self):
        return 1 - stats.f.cdf(self.f_statistic, len(self.independent_variables) - 1, len(self.df) - len(self.independent_variables))
    
    def get_t_statistics(self):
        return self.coefficients / np.sqrt(np.diag(np.linalg.inv(self.df[self.independent_variables].T.dot(self.df[self.independent_variables]))))
    
    def get_t_p_values(self):
        return 2 * (1 - stats.t.cdf(np.abs(self.t_statistics), len(self.df) - len(self.independent_variables)))
    
    def get_summary(self):
        summary = pd.DataFrame
        return summary


#create a regression model for the wine dataset
wine_model = RegressionModel(df, "Alcohol", ["Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",  "Flavanoids",
                             "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"])

wine_model.get_coefficients()

wine_model.get_intercept()

wine_model.get_predicted()

validacion = pd.DataFrame({'Actual': df['Alcohol'].values, 'Predicted': wine_model.get_predicted(
), 'Diferencia': df['Alcohol'] - wine_model.get_predicted(), 'Residuals': wine_model.get_residuals()})
l = validacion.head(25)
l


# creamos un gráfico de barras con el dataframe que contiene nuestros datos actuales y de predicción
l.plot.bar(rot=0, figsize=(20, 10))
# indicamos el título del gráfico
plt.title("Comparación ")
# indicamos la etiqueta del eje de las x
plt.xlabel("Muestra")
# indicamos la etiqueta del eje de las y, la cantidad de calorías
plt.ylabel("Cantidad de Alcohol")
plt.show()  # desplegamos el gráfico
