import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import linear_model
import statsmodels.api as sm

# Import the entire data sets
math_df = pd.read_csv("../data/student-mat.csv", sep=";")
portugese_df = pd.read_csv("../data/student-por.csv", sep=";")

# Merge dataframes
df = math_df.append(portugese_df)
df.replace(["Yes", "No"], [1, 0])

# Parametere som har lineære sammenhenger med G3 (i både math og portugese)
linear_params = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'activities', 'higher', 'internet', 'romantic', 'Dalc', 'absences'
    ]

# Grafer som viser sammenhengen mellom G3 og parameterne

X = df[linear_params]
y = df["G3"]

for col in X:
    fig = go.Figure(data=go.Scatter(x=X[col], y=y, mode='markers'))
    fig.update_layout(title=col)
    fig.show()


def linear_prediction_model():
    X = df[['studytime', 'failures', 'Dalc', 'absences']]
    y = df["G3"]

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    test_y = [[3, 2, 1, 24]]
    prediction_result  = regr.predict(test_y)[0]
    print(prediction_result)
    return 0
linear_prediction_model()