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
data_url = 'https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/life-exp/data/life-expectancy.csv'

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

"""
REGRESSION PREDICTION MODEL
The function linear_prediction_model makes a predictive model using linear regression and
data from the data set.
"""

# Import the entire data sets
df = pd.read_csv(data_url, sep=",")
df.drop(['Country', 'Year', 'Status'], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]

all_parameters = [
    'Adult mortality','Infant deaths','Alcohol','Percentage expenditure', 'Hepatitis B',
    'Measles','BMI','Under five deaths','Polio','Total expenditure', 'Diphtheria','HIV/AIDS',
    'GDP','Population','Thinness 1 to 19','Thinness 5 to 9','HDI income','Schooling'
    ]

# Parameters with linearity
parameter_list = ['Schooling', 'HDI income', 'Adult mortality']

def linear_prediction_model(parameter_list, selected_values):
    X = df_regr[parameter_list].round(2)
    y = df_regr['Life expectancy'].round(2)

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    prediction_result  = regr.predict([selected_values])[0]
    return round(prediction_result, 2)


""" LAYOUT """

app.layout = html.Div([
    html.Div([
        html.H2('Life expectancy prediction'),
        html.Div(
            "This is a tool for predicting the life expectancy of the population in your country. The prediction"
            + "model is built from a multiple linear regression on a data set from WHO."
        ),
        html.H5("Schooling"),
        html.Div("Select the average number of schooling years:"),
        dcc.Dropdown(
            id='schooling',
            options=[{'label': i, 'value': i} for i in range(0,22)],
            value=0
        ),
        html.H5("HDI"),
        html.Div("Select your Human Development Index (HDI) in terms of income composition of resources:"),
        dcc.Slider(
            id='hdi-income',
            min=0,
            max=1,
            step=0.01,
            value=0,
            marks={0:'0', 0.1:'0.1', 0.2:'0.2', 0.3:'0.3', 0.4:'0.4', 0.5:'0.5', 0.6:'0.6', 0.7:'0.7', 0.8:'0.8', 0.9:'0.9', 1:'1'}
        ),
        html.H5("Adult mortality"),
        html.Div("Select the adult mortality rate per 1000 population: "),
        dcc.Slider(
            id='adult-mortality',
            min=0,
            max=1000,
            step=10,
            value=0,
            marks={0:'0', 100:'100', 200:'200', 300:'300', 400:'400', 500:'500', 600:'600', 700:'700', 800:'800', 900:'900', 1000:'1000'}
        ),
        html.H4("The predicted life expectancy in years is: "),
        html.H4(id='prediction'),
    ], className="six columns"),
    
    html.Div([
        html.H2('Correlation between a selected parmeter and the life expectancy'),
        html.Div(
            "This is a visualization of the raw data from the data sets."
        ),
        dcc.Dropdown(
            id='select-parameter',
            options=[{'label': i, 'value': i} for i in all_parameters],
            value='AdultMortality'
        ),
        html.Div(dcc.Graph(id='correlation-plot'))
    ], className="six columns")
])

"""
CALLBACK FUNCTIONS
Thede function changes the values of the parameters used in the prediction, and
changes the plots
"""

@app.callback(
    Output('prediction', 'children'),
    [
        Input('schooling', 'value'),
        Input('hdi-income', 'value'),
        Input('adult-mortality', 'value')
    ]
)
def get_prediction_result(schooling, hdi_income, adult_mortality):
    selected_values = [schooling, hdi_income, adult_mortality]
    predicted_grade = linear_prediction_model(parameter_list, selected_values)
    return predicted_grade

@app.callback(
    Output('correlation-plot', 'figure'),
    [
        Input('select-parameter', 'value')
    ]
)
def make_correlation_graph(select_parameter):
    X = df_regr[select_parameter].round(2)
    y = df_regr['Life expectancy'].round(2)
    fig = go.Figure(data=go.Scatter(x=X, y=y, mode='markers'))
    fig.update_layout(
        yaxis_title="Life expectancy",
        xaxis_title="{select_parameter}"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

