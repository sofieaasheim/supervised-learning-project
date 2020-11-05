import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import linear_model
import statsmodels.api as sm

# Import the entire data sets
math_df = pd.read_csv("../data/student-mat.csv", sep=";")
portugese_df = pd.read_csv("../data/student-por.csv", sep=";")

# Fjern dustekolonner
math_df.drop(['sex', 'address', 'school', 'Mjob', 'Fjob'], axis=1, inplace=True)
portugese_df.drop(['sex', 'address', 'school', 'Mjob', 'Fjob'], axis=1, inplace=True)

# Gjør om alle yes/no til 0/1
math_df.replace(('yes', 'no'), (1, 0), inplace=True)
portugese_df.replace(('yes', 'no'), (1, 0), inplace=True)

# Fiks 'famsize' kolonnen
math_df.replace(('LE3', 'GT3'), (0, 1), inplace=True)
portugese_df.replace(('LE3', 'GT3'), (0, 1), inplace=True)

# Fiks 'Pstatus' kolonnen
math_df.replace(('A', 'T'), (0, 1), inplace=True)
portugese_df.replace(('A', 'T'), (0, 1), inplace=True)

# Fiks 'reason' kolonnen
math_df.replace(('home', 'reputation', 'course', 'other'), (1, 2, 3, 4), inplace=True)
portugese_df.replace(('home', 'reputation', 'course', 'other'), (1, 2, 3, 4), inplace=True)

# Fiks 'guardian' kolonnen
math_df.replace(('mother', 'father', 'other'), (1, 2, 3), inplace=True)
portugese_df.replace(('mother', 'father', 'other'), (1, 2, 3), inplace=True)

# Parametere som har lineære sammenhenger med G3 (i både math og portugese)
linear_params = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
    'activities', 'higher', 'internet', 'romantic', 'Dalc', 'absences'
    ]

# Divide into parameters and responses
math_parameter_df = math_df[linear_params]
math_response_df = math_df[["G1", "G2", "G3"]]

por_parameter_df = portugese_df[linear_params]
por_response_df = portugese_df[["G1", "G2", "G3"]]

# Linear regression test
X = portugese_df[['studytime', 'failures', 'Dalc']]
y = portugese_df["G3"]

print(portugese_df["failures"])

# Grafer som viser sammenhengen mellom G3 og parameterne
"""
for col in X:
    fig = go.Figure(data=go.Scatter(x=X[col], y=y, mode='markers'))
    fig.update_layout(title=col)
    fig.show()
"""
"""
# Lineær regresjon med statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)
"""

def linear_prediction_model():
    X = portugese_df[['studytime', 'failures', 'Dalc', 'absences']]
    y = portugese_df["G3"]

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    test_y = [[1, 1, 1, 0]]
    prediction_result  = regr.predict(test_y)[0]
    print(prediction_result)
    return 0
linear_prediction_model()