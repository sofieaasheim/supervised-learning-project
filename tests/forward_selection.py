import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiple_regression import multiple_regression

""" IMPORT AND PREPROCESS THE DATA """

data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"

# Import the entire data set
df = pd.read_csv(data_url, sep=",")

# Remove non-relevant parameters for the regression and remove all non-finite values such as NaN and +/- infinity
df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]


initial_parameters = [
    "AdultMortality",
    "InfantDeaths",
    "Alcohol",
    "PercentageExpenditure",
    "HepatitisB",
    "BMI",
    "Polio",
    "TotalExpenditure",
    "HIVAIDS",
    "Diphtheria",
    "Thinness1_19",
    "Income",
    "Schooling",
]

#X = df_regr[parameter_list]
#y = df_regr["LifeExpectancy"]
print(multiple_regression(df_regr, initial_parameters[0]))
print(multiple_regression(df_regr, initial_parameters[1]))
print(multiple_regression(df_regr, initial_parameters[2]))
print(multiple_regression(df_regr, initial_parameters[3]))
print(multiple_regression(df_regr, initial_parameters[4]))
print(multiple_regression(df_regr, initial_parameters[5]))
print(multiple_regression(df_regr, initial_parameters[6]))
print(multiple_regression(df_regr, initial_parameters[7]))
print(multiple_regression(df_regr, initial_parameters[8]))
print(multiple_regression(df_regr, initial_parameters[9]))
print(multiple_regression(df_regr, initial_parameters[10]))
print(multiple_regression(df_regr, initial_parameters[11]))
print(multiple_regression(df_regr, initial_parameters[12]))











   
