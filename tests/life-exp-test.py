import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import linear_model
import statsmodels.api as sm

# Import the entire data sets
df = pd.read_csv("../data/life-expectancy.csv", sep=",")

# Dependent and independent variables
X = df.drop(['Country', 'Year', 'Status', 'LifeExpectancy'], axis=1)
y = df['LifeExpectancy']

# Lage korrelasjonsplott for alle parameterne
for col in X:
    fig = go.Figure(data=go.Scatter(x=X[col], y=y, mode='markers'))
    fig.update_layout(title=col)
    fig.show()