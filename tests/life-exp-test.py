import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import linear_model
import statsmodels.api as sm

# Import the entire data sets
df = pd.read_csv("../data/life-expectancy.csv", sep=",")
df.drop(['Country', 'Year', 'Status'], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]

# Dependent and independent variables
X = df_regr[['Schooling', 'Income', 'AdultMortality']].round(decimals=2)
y = df_regr['LifeExpectancy'].round(decimals=2)
print(np.any(np.isnan(df_regr))) #and gets False
print(np.all(np.isfinite(df_regr)))


# Make correlation plots for all parameters
"""for col in X:
    fig = go.Figure(data=go.Scatter(x=X[col], y=y, mode='markers'))
    fig.update_layout(title=col)
    fig.show()"""

"""Parameters showing linearity:
AdultMortality, InfantDeaths, UnderFiveDeaths, Polio, Diphteria, HIVAIDS, GDP, Income, Schooling
"""

# Regression test
def linear_prediction_model():
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    test_y = [[10, 200, 300]]
    prediction_result  = regr.predict(test_y)[0]
    print(prediction_result)
    return 0
linear_prediction_model()