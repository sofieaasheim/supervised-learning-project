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

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Import the entire data sets
df = pd.read_csv(data_url, sep=",")
df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]

all_parameters = [
    "AdultMortality",
    "InfantDeaths",
    "Alcohol",
    "PercentageExpenditure",
    "HepatitisB",
    "Measles",
    "BMI",
    "UnderFiveDeaths",
    "Polio",
    "TotalExpenditure",
    "Diphtheria",
    "HIVAIDS",
    "GDP",
    "Population",
    "Thinness1_19",
    "Thinness5_9",
    "Income",
    "Schooling",
]
parameter_names = [
    "Adult mortality (per 1000 population)",
    "Infant deaths (per 1000 population)",
    "Alcohol consumption (pure alcohol in litres per capita)",
    "Percentage expenditure on health of GDP per capita (%)",
    "Hepatitis B immunization coverage among 1-year olds (%)",
    "Measles cases (per 1000 population)",
    "Average BMI of entire population",
    "Under-5 deaths (per 1000 population)",
    "Polio immunization coverage among 1-year olds (%)",
    "Total expenditure on health (% of total government expenditure)",
    "Diphtheria (DTP3) immunzation coverage among 1-year olds (%)",
    "HIV/AIDS deaths (per 1000 HIV/AIDS live births 0-4 years)",
    "GDP per capita (USD)",
    "Population of the country",
    "Thinness age 1 to 19 (%)",
    "Thinness age 5 to 9 (%)",
    "Human Development Index (HDI) (0-1)",
    "Number of years of schooling",
]

# Parameters with linearity
parameter_list = ["Schooling", "Income", "AdultMortality"]

"""
REGRESSION PREDICTION MODEL
The function linear_prediction_model makes a predictive model using linear regression and
data from the data set.
"""


def linear_prediction_model(parameter_list, selected_values):
    X = df_regr[parameter_list].round(2)
    y = df_regr["LifeExpectancy"].round(2)

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    prediction_result = regr.predict([selected_values])[0]
    return round(prediction_result, 2)


"""
CORRELATION MATRIX
The function make_correlation_matrix makes a heatmap showing all correlations
between the parameters in terms of the Pearson correlation coefficient.
"""


def make_correlation_matrix(df_regr):
    correlation_df = df_regr[all_parameters].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_df,
            x=list(correlation_df.columns),
            y=list(correlation_df.columns),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        height=500,
        margin={"t": 20},
        font_family="Helvetica",
    )
    return fig


""" LAYOUT """

app.layout = html.Div(
    [
        html.Br(),
        html.Br(),
        html.Div([], className="one column"),
        html.Div(
            [
                html.H2("Life expectancy prediction"),
                html.Div(
                    [
                        "This is a tool for predicting the life expectancy of the population in your country. The prediction "
                        + "model is built from a multiple linear regression on a data set from WHO found ",
                        html.A(
                            "here",
                            href="https://www.kaggle.com/kumarajarshi/life-expectancy-who",
                        ),
                        ".",
                        html.Br(),
                        html.Br(),
                        "After performing many tests, the parameters below turned out to be the most "
                        + "statistically significant parameter combination for this predictive model. ",
                        html.Br(),
                        html.Br(),
                        "(Jeg tenker vi oppdaterer parameterne ut i fra hvilke resultater vi får av regresjonen, "
                        + "de under er bare et foreløpig forslag.)",
                    ]
                ),
                html.Br(),
                html.H5("Schooling"),
                html.Div("Select the average number of schooling years:"),
                dcc.Dropdown(
                    id="schooling",
                    options=[{"label": i, "value": i} for i in range(0, 22)],
                    value=0,
                ),
                html.Br(),
                html.H5("HDI"),
                html.Div(
                    "Select your Human Development Index (HDI) in terms of income composition of resources:"
                ),
                dcc.Slider(
                    id="hdi-income",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0,
                    marks={
                        0: "0",
                        0.1: "0.1",
                        0.2: "0.2",
                        0.3: "0.3",
                        0.4: "0.4",
                        0.5: "0.5",
                        0.6: "0.6",
                        0.7: "0.7",
                        0.8: "0.8",
                        0.9: "0.9",
                        1: "1",
                    },
                ),
                html.Br(),
                html.H5("Adult mortality"),
                html.Div("Select the adult mortality rate per 1000 population: "),
                dcc.Slider(
                    id="adult-mortality",
                    min=0,
                    max=1000,
                    step=10,
                    value=0,
                    marks={
                        0: "0",
                        100: "100",
                        200: "200",
                        300: "300",
                        400: "400",
                        500: "500",
                        600: "600",
                        700: "700",
                        800: "800",
                        900: "900",
                        1000: "1000",
                    },
                ),
                html.Br(),
                html.H4("The predicted life expectancy in years is: "),
                html.H3(id="prediction"),
            ],
            style={"backgroundColor": "#F5F5F5", "padding": "0px 20px 20px 20px"},
            className="five columns",
        ),
        html.Div(
            [
                html.H2("Raw data visualizations"),
                html.H4(
                    "Correlation between a selected parameter and the life expectancy"
                ),
                html.Div("Select a parameter:"),
                dcc.Dropdown(
                    id="select-parameter",
                    options=[
                        {"label": i, "value": j}
                        for i, j in zip(parameter_names, all_parameters)
                    ],
                    value="AdultMortality",
                ),
                html.Div(dcc.Graph(id="correlation-plot")),
                html.H4("Correlations between all parameters (independent variables)"),
                html.Div(
                    dcc.Graph(
                        id="correlation-matrix", figure=make_correlation_matrix(df_regr)
                    )
                ),
            ],
            className="six columns",
        ),
    ]
)

"""
CALLBACK FUNCTIONS
Thede function changes the values of the parameters used in the prediction, and
changes the plots
"""


@app.callback(
    Output("prediction", "children"),
    [
        Input("schooling", "value"),
        Input("hdi-income", "value"),
        Input("adult-mortality", "value"),
    ],
)
def get_prediction_result(schooling, hdi_income, adult_mortality):
    selected_values = [schooling, hdi_income, adult_mortality]
    predicted_grade = linear_prediction_model(parameter_list, selected_values)
    return predicted_grade


@app.callback(
    Output("correlation-plot", "figure"), [Input("select-parameter", "value")]
)
def make_correlation_graph(select_parameter):
    X = df_regr[select_parameter].round(2)
    y = df_regr["LifeExpectancy"].round(2)
    fig = go.Figure(data=go.Scatter(x=X, y=y, mode="markers"))
    fig.update_layout(
        yaxis_title="Life expectancy",
        xaxis_title=f"{select_parameter}",
        height=400,
        margin={"t": 20, "b": 20},
        font_family="Helvetica",
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
