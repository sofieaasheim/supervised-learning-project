import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import linear_model
import statsmodels.api as sm
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

"""
REGRESSION PREDICTION MODEL
The function linear_prediction_model makes a predictive model using linear regression and
data from the data sets.
"""

# Import the entire data sets
math_df = pd.read_csv("./data/student-mat.csv", sep=";")
portugese_df = pd.read_csv("./data/student-por.csv", sep=";")

parameter_list =['studytime', 'failures', 'Dalc', 'absences']

def linear_prediction_model(parameter_list, selected_values):
    X = portugese_df[parameter_list]
    y = portugese_df["G3"]

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    prediction_result  = regr.predict([selected_values])[0]
    return round(prediction_result, 2)


""" LAYOUT """
app.layout = html.Div([
    html.H2('Predict your final grade'),
    html.Div(
        "This is a tool for predicting your final grade based on different factors. " +
        "The grades goes from 0 to 20, which is the Portugese grading system."
    ),
    html.H5("Studytime"),
    html.Div("Select your weekly study time. 1: less than 2 hours, 2: 2-5 hours, 3: 5 to 10 hours or 4: more than 10 hours"),
    dcc.Dropdown(
        id='studytime',
        options=[{'label': i, 'value': i} for i in [1, 2, 3, 4]],
        value=1
    ),
    html.H5("Failures"),
    html.Div("Select the number of your past class failures."),
    dcc.Dropdown(
        id='failures',
        options=[{'label': i, 'value': i} for i in [0, 1, 2, 3, 4]],
        value=0
    ),
    html.H5("Alcohol consumption"),
    html.Div("Rate your workday alchol consumption. 1 = very low, 5 = very high."),
    dcc.Dropdown(
        id='Dalc',
        options=[{'label': i, 'value': i} for i in [1, 2, 3, 4, 5]],
        value=1
    ),
    html.H5("Absences"),
    html.Div("Select your number of absences the last year"),
    dcc.Dropdown(
        id='absences',
        options=[{'label': i, 'value': i} for i in range(0, 94)],
        value=0
    ),
    html.H4("The predicted grade is: "),
    html.H4(id='prediction')
])


"""
CALLBACK FUNCTION
This function changes the values of the parameters used in the prediction
"""

@app.callback(
        Output('prediction', 'children'),
    [
        Input('studytime', 'value'),
        Input('failures', 'value'),
        Input('Dalc', 'value'),
        Input('absences', 'value'),
    ]
)
def get_prediction_result(studytime, failures, Dalc, absences):
    selected_values = [studytime, failures, Dalc, absences]
    predicted_grade = linear_prediction_model(parameter_list, selected_values)
    return predicted_grade


if __name__ == '__main__':
    app.run_server(debug=True)

