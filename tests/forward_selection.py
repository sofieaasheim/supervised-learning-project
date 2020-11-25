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

# Look at the p-values for all the initial parameters
for i in range(12):
    print(multiple_regression(df_regr, initial_parameters[i]))

# As we can see all the p-values are microspopic. 
# Therefore this is not an optimal method for this data set. 









   
