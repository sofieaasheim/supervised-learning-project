import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn import linear_model
import statsmodels.api as sm
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

server = app.server

# Import the entire data sets
df = pd.read_csv(data_url, sep=",")
df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]

all_parameters = [
    "AdultMortality",
    "InfantDeaths",
    "Alcohol",
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
    "Hepatitis B immunization coverage among 1-year olds (%)",
    "Measles cases (per 1000 population)",
    "Average BMI of entire population",
    "Under-5 deaths (per 1000 population)",
    "Polio immunization coverage among 1-year olds (%)",
    "Total expenditure on health (% of total expenditure)",
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
parameter_list = [
    "Schooling",
    "Income",
    "HIVAIDS",
    "BMI",
    "AdultMortality",
    "Alcohol",
    "Diphtheria",
    "GDP"
]

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
        dbc.Container(
            [
                html.Br(),
                html.Br(),
                html.H1("MULTIPLE REGRESSION", style={"textAlign": "center"}),
                html.Br(),
                dbc.Col(
                    [   # BOKS FOR KNAPPER
                        dbc.Row(
                            dbc.Col(
                                html.Div(
                                    [
                                        dbc.Button(
                                            "What is machine learning?",
                                            id="open-1",
                                            className="mr-1",
                                            color="success"
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Machine learning"),
                                                dbc.ModalBody([
                                                    "You have probably heard about machine learning before. People say it is the future and they talk about how cool it is, "
                                                    + "but what is actually machine learning? And why is it so cool? Machine learning is a type of artificial intelligence "
                                                    + "that makes it possible for a computer to automatically improve through experience. Simply put, this actually means "
                                                    + "that the computer is able to ",
                                                    html.B("learn "), 
                                                    "without being explicitly programmed to do so. Cool.",
                                                    html.Br(), html.Br(),
                                                    "This learning process begins with observations of data – i.e. that the computer looks at examples or patterns in the data, "
                                                    + "or is given some kind of instructions. Then, the computer will use this information to make better decisions in the future "
                                                    + "based on these experiences. The goal of this is to make the computer’s learning process automatic.",
                                                    html.Br(), html.Br(),
                                                    "There are many types of machine learning algorithms and methods, and these are usually split into three main categories:",
                                                    html.Br(), html.Br(),
                                                    html.B("Supervised learning"), html.Br(),
                                                    "Algorithms within the field of supervised learning used labeled examples to learn from. These labeled examples are called "
                                                    + "«training datasets», and the algorithm uses this data to make predictions about the output values. The machine learning method "
                                                    + "called multiple regression, which is used in this project, is from the field of supervised learning.",
                                                    html.Br(), html.Br(),
                                                    html.B("Unsupervised learning"), html.Br(),
                                                    "In these algorithms, the information used to train the model is neither classified nor labeled. This means that the computer need "
                                                    + "to look for a hidden structure or a pattern in the data. It explores the data on its own without any «help» from examples or instructions.",
                                                    html.Br(), html.Br(),
                                                    html.B("Reinforcement learning"), html.Br(),
                                                    "The algoritmhs within the field of reinforcement learning is a bit different from the algorithms in the two categories above. Here, "
                                                    + "the computer interacts with the environment by producing actions, and from this, finding out what action is the «best» one. The "
                                                    + "prodecure used trial and error, and gains rewards or penalties based on this. This allows for the computer to automatically determine the ideal behavior.",
                                                    html.Br(), html.Br(),
                                                    html.I("The figure below shows the three main categories of machine learning, and the most common methods and algorithms within each of the categories."),
                                                    html.Br(), html.Br(),
                                                    html.Img(src="https://wordstream-files-prod.s3.amazonaws.com/s3fs-public/styles/simple_image/public/images/machine-learning1.png?SnePeroHk5B9yZaLY7peFkULrfW8Gtaf&itok=yjEJbEKD"
                                                    , style={"width": "100%"}),
                                                    html.Br()
                                                ]),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close",
                                                        id="close-1",
                                                        className="ml-auto",
                                                    )
                                                ),
                                            ],
                                            id="modal-1",
                                            size="lg",
                                            scrollable=True
                                        ),
                                        dbc.Button(
                                            "The data set",
                                            id="open-2",
                                            className="mr-1",
                                            color="info"
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("The data behind the model"),
                                                dbc.ModalBody([
                                                    "The data set contains data from 193 countries, and for each country there are data from the years 2000-2015. "
                                                    + "The data set contains 2938 rows and 22 columns, where one of the columns is the life expectancy (response) and "
                                                    + "the 21 remaining columns are the predicting variables (parameters).",
                                                    html.Br(), html.Br(),
                                                    "The parameters includes factors that possibly affects the life expectancy in a country, such as demographic "
                                                    + "variables, income composition and mortality rates. Some examples of parameters are: the number of infant deaths, "
                                                    + "alcohol consumption, average Body Mass Index (BMI) and Gross Domestic Product (GDP). Two of the parameters have"
                                                    + "string values, while the rest have numerical values."
                                                ]),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close",
                                                        id="close-2",
                                                        className="ml-auto",
                                                        
                                                    )
                                                ),
                                            ],
                                            id="modal-2",
                                            size="lg",
                                            scrollable=True
                                        ),
                                        dbc.Button(
                                            "Regression",
                                            id="open-3",
                                            className="mr-1",
                                            color="warning"
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Regression"),
                                                dbc.ModalBody([
                                                    "Regression is a very helpful statistical method used to find trends in data. So why is this helpful? Well, "
                                                    + "imagine that you guess that there is a connection between how many hours you study and how well you do on the exam; "
                                                    + "regression analysis can actually help you quantify this. What more, if you make a regression model, it can predict "
                                                    + "what grade you will get on the next exam by how many hours you study. The accuracy of the prediction will depend upon "
                                                    + "how good your model is.",
                                                    html.Br(), html.Br(),
                                                    "There are several different forms of regression. A linear regression model is a statistical model and one of the basic "
                                                    + "building blocks of machine learning. The model attempts to model the relationship between two variables by fitting a "
                                                    + "linear equation to observed data. For the model to be of interest there has to be some significant association between "
                                                    + "the two variables. The equation for the linear regression line takes the form",
                                                    html.Br(), html.Br(),
                                                    html.I("y"), " = B", html.I("x "), "+ A,",
                                                    html.Br(), html.Br(),
                                                    "where y is the response and x is the parameter. B is a coefficient gives the slope of the line and A is the intercept, "
                                                    + "that is the value of y when x = 0.",
                                                    html.Br(), html.Br(),
                                                    "To predict life expectancy from multiple variables as we have done in this project, one needs to use multiple linear regression. "
                                                    + "Multiple linear regression explains the relationship between one dependent variable, a response, and two or more independent "
                                                    + "variables, parameters.  Even though it is more advanced than linear regression it builds on the same principles. Take a look at "
                                                    + "section 3.2.1 of our paper for a theoretical explanation."
                                                ]),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close",
                                                        id="close-3",
                                                        className="ml-auto",
                                                    )
                                                ),
                                            ],
                                            id="modal-3",
                                            size="lg",
                                            scrollable=True
                                        ),
                                        dbc.Button(
                                            "Correlation",
                                            id="open-4",
                                            className="mr-1",
                                            color="danger"
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Correlation"),
                                                dbc.ModalBody([
                                                    "If you want to do a multiple regression on a data set, you first need to have an understanding of "
                                                    + "what correlation means. This is imporant in the data preprocessing phase – in other words ",
                                                    html.I("before "), 
                                                    "executing the regression. The reason behind this is that for a multiple regression to work properly, "
                                                    + "all of the parameters included needs to show some sort of linear relationship with the response.",
                                                    html.Br(), html.Br(),
                                                    "Correlation means association, and in other words, it is a measure of how much two variables are "
                                                    + "related to each other. This can be visualized by plotting two variables (in this case, one parameter "
                                                    + "and the response) in a scatter plot, and see if the points lay close to a line.",
                                                    html.Br(), html.Br(),
                                                    html.B("Types of correlation"), html.Br(),
                                                    "Correlations can be either positive, negative or none.",
                                                    html.Br(), html.Br(),
                                                    "If there is a positive correlation, both of the variables move in the same direction. "
                                                    + "An example of positive correlation can be height and weight – tall people tend to be heavier.",
                                                    html.Br(), html.Br(),
                                                    "If the correlation is negative, an increase in one variable is associated with a decrease "
                                                    + "in the other variable. For example, look at height above the sea level and the temperature: the "
                                                    + "higher above sea level (increase), the lower the temperature (decrease)",
                                                    html.Br(), html.Br(),
                                                    "In many cases, two variables show no correlation at all. This means that there is no "
                                                    + "relationship between them. This can for example be the relationship between the amount of "
                                                    + "burgers eaten and the level of intelligence (or…?).", html.Br(),html.Br(),
                                                    html.Img(src="https://www.biologyforlife.com/uploads/2/2/3/9/22392738/correlation_1.jpg?688"
                                                    , style={"width": "60%", "textAlign": "center"}), html.Br(), html.Br(),
                                                    "When performing a multiple regression, you should only include parameters that are correlated with the response.",
                                                    html.Br(), html.Br(),
                                                    html.B("Correlation coefficients"), html.Br(),
                                                    "In the graph under the section “Correlations between all parameters” on this website, all the parameter’s correlation coefficients are visualized.",
                                                    html.Br(), html.Br(),
                                                    "This is a number between -1 and 1, and it describes how much the parameters are correlated with each other. "
                                                    + "The further the number is away from 0, the higher the correlation is. A value of +1 indicated a total positive "
                                                    + "linear correlation, 0 is no correlation and -1 is a total negative linear correlation.",
                                                    html.Br(), html.Br(),
                                                    "In this project, we use the Pearson correlation coefficient. If you want to know more about this, check out the ",
                                                    html.A("Wikipedia", href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient"),
                                                    " page."
                                                ]),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close",
                                                        id="close-4",
                                                        className="ml-auto",
                                                    )
                                                ),
                                            ],
                                            id="modal-4",
                                            size="lg",
                                            scrollable=True
                                        ),
                                        dbc.Button(
                                            "What do the results tell us?",
                                            id="open-5",
                                            className="mr-1",
                                            color="primary"
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("What do the results tell us?"),
                                                dbc.ModalBody([
                                                    "Sett inn tekst her"
                                                ]),
                                                dbc.ModalFooter(
                                                    dbc.Button(
                                                        "Close",
                                                        id="close-5",
                                                        className="ml-auto",
                                                    )
                                                ),
                                            ],
                                            id="modal-5",
                                            size="lg",
                                            scrollable=True
                                        ),
                                    ]
                                ),
                            ), style={"margin": "auto", 'textAlign':'center'},
                        ),
                        html.Br(),
                        # BOKS FOR PREDICTION
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H2("Life expectancy prediction"),
                                            html.Div(
                                                [
                                                    "This is a tool for predicting the average life expectancy of the population in your country. "
                                                    + "The prediction model is built from a multiple linear regression on a data set from WHO found ",
                                                    html.A(
                                                        "here",
                                                        href="https://www.kaggle.com/kumarajarshi/life-expectancy-who",
                                                    ),
                                                    ".",
                                                    html.Br(),
                                                    html.Br(),
                                                    "When applying the backwards elimination method, the parameters below turned out to be the most "
                                                    + "statistically significant parameters from the data set for this predictive model. ",
                                                    html.Br(),
                                                    html.Br(),
                                                    "For more information about the project, check out our ",
                                                    html.A(
                                                        "GitHub repository",
                                                        href="https://github.com/sofieaasheim/supervised-learning-project",
                                                    ),
                                                    ".",
                                                ]
                                            ),
                                            html.Br(),
                                            html.H5("Schooling"),
                                            html.Div(
                                                "Select the average number of schooling years:"
                                            ),
                                            dcc.Dropdown(
                                                id="schooling",
                                                options=[
                                                    {"label": i, "value": i}
                                                    for i in range(0, 22)
                                                ],
                                                value=0,
                                            ),
                                            html.Br(),
                                            html.H5("HDI"),
                                            html.Div(
                                                "Select the Human Development Index (HDI) in terms of income composition of resources:"
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
                                            html.H5("HIV/AIDS"),
                                            html.Div(
                                                "Select the amount of deaths per 1000 live births HIV/AIDS (0-4 years): "
                                            ),
                                            dcc.Slider(
                                                id="hivaids",
                                                min=0,
                                                max=50,
                                                step=5,
                                                value=0,
                                                marks={
                                                    0: "0",
                                                    5: "5",
                                                    10: "10",
                                                    15: "15",
                                                    20: "20",
                                                    25: "25",
                                                    30: "30",
                                                    35: "35",
                                                    40: "40",
                                                    45: "45",
                                                    50: "50",
                                                },
                                            ),
                                            html.Br(),
                                            html.H5("BMI"),
                                            html.Div(
                                                "Select the average Body Mass Index (BMI) of the population: "
                                            ),
                                            dcc.Slider(
                                                id="bmi",
                                                min=0,
                                                max=80,
                                                step=1,
                                                value=0,
                                                marks={
                                                    0: "0",
                                                    10: "10",
                                                    20: "20",
                                                    30: "30",
                                                    40: "40",
                                                    50: "50",
                                                    60: "60",
                                                    70: "70",
                                                    80: "80",
                                                },
                                            ),
                                            html.Br(),
                                            html.H5("Adult mortality"),
                                            html.Div(
                                                "Select the adult mortality rate (per 1000 population): "
                                            ),
                                            dcc.Slider(
                                                id="adult-mortality",
                                                min=0,
                                                max=800,
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
                                                },
                                            ),
                                            html.Br(),
                                            html.H5("Alcohol"),
                                            html.Div(
                                                "Select the alcohol consumption recorded per capita (15+) in litres of pure alcohol: "
                                            ),
                                            dcc.Dropdown(
                                                id="alcohol",
                                                options=[
                                                    {"label": i, "value": i}
                                                    for i in range(0, 19)
                                                ],
                                                value=0,
                                            ),
                                            html.Br(),
                                            html.H5("Diphtheria"),
                                            html.Div(
                                                "Select the diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%): "
                                            ),
                                            dcc.Slider(
                                                id="diphtheria",
                                                min=0,
                                                max=100,
                                                step=1,
                                                value=0,
                                                marks={
                                                    0: "0",
                                                    10: "10",
                                                    20: "20",
                                                    30: "30",
                                                    40: "40",
                                                    50: "50",
                                                    60: "60",
                                                    70: "70",
                                                    80: "80",
                                                    90: "90",
                                                    100: "100",
                                                },
                                            ),
                                            html.Br(),
                                            html.H5("GDP"),
                                            html.Div(
                                                "Select the Gross Domestic Product per capita (in USD): "
                                            ),
                                            dcc.Slider(
                                                id="gdp",
                                                min=0,
                                                max=100000,
                                                step=100,
                                                value=0,
                                                marks={
                                                    0: "0",
                                                    10000: "10000",
                                                    20000: "20000",
                                                    30000: "30000",
                                                    40000: "40000",
                                                    50000: "50000",
                                                    60000: "60000",
                                                    70000: "70000",
                                                    80000: "80000",
                                                    90000: "90000",
                                                    10000: "100000",
                                                },
                                            ),
                                            html.Br(),
                                            html.H4(
                                                "The predicted life expectancy in years is: "
                                            ),
                                            html.H3(id="prediction"),
                                        ],
                                        style={
                                            "backgroundColor": "#E6E6E6",
                                            "padding": "20px 20px 20px 20px",
                                        },
                                    )
                                ),
                                # BOKS FOR RAW DATA
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H2("Raw data visualizations"),
                                            html.H4(
                                                "Correlation between the parameters and the life expectancy"
                                            ),
                                            html.Div(
                                                "The visualization below shows a scatter plot of a selected parameter "
                                                + "and the life expectancy. Select a parameter:"
                                            ),
                                            html.Br(),
                                            dcc.Dropdown(
                                                id="select-parameter",
                                                options=[
                                                    {"label": i, "value": j}
                                                    for i, j in zip(
                                                        parameter_names, all_parameters
                                                    )
                                                ],
                                                value="AdultMortality",
                                            ),
                                            html.Div(dcc.Graph(id="correlation-plot")),
                                            html.Br(),
                                            html.H4(
                                                "Correlations between all parameters"
                                            ),
                                            html.Div(
                                                [
                                                    "This visualization shows a heat map of all the ",
                                                    html.A(
                                                        "Pearson correlation coefficients",
                                                        href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",
                                                    ),
                                                    " of the parameters in the data set.",
                                                ]
                                            ),
                                            html.Div(
                                                dcc.Graph(
                                                    id="correlation-matrix",
                                                    figure=make_correlation_matrix(
                                                        df_regr
                                                    ),
                                                )
                                            ),
                                        ], style={"padding": "20px 20px 20px 20px", "backgroundColor": "white"}
                                    )
                                ),
                            ]
                        ),
                    ]
                ),
                html.Br(),
                html.Br(),
            ]
        )
    ], style={"backgroundColor": "#F7F7F7"}
)

"""
CALLBACK FUNCTIONS
Thede function changes the values of the parameters used in the prediction, and
changes the plots
"""

def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

app.callback(
    Output("modal-1", "is_open"),
    [Input("open-1", "n_clicks"), Input("close-1", "n_clicks")],
    [State("modal-1", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal-2", "is_open"),
    [Input("open-2", "n_clicks"), Input("close-2", "n_clicks")],
    [State("modal-2", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal-3", "is_open"),
    [Input("open-3", "n_clicks"), Input("close-3", "n_clicks")],
    [State("modal-3", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal-4", "is_open"),
    [Input("open-4", "n_clicks"), Input("close-4", "n_clicks")],
    [State("modal-4", "is_open")],
)(toggle_modal)

app.callback(
    Output("modal-5", "is_open"),
    [Input("open-5", "n_clicks"), Input("close-4", "n_clicks")],
    [State("modal-5", "is_open")],
)(toggle_modal)


@app.callback(
    Output("prediction", "children"),
    [
        Input("schooling", "value"),
        Input("hdi-income", "value"),
        Input("hivaids", "value"),
        Input("bmi", "value"),
        Input("adult-mortality", "value"),
        Input("alcohol", "value"),
        Input("diphtheria", "value"),
        Input("gdp", "value")
    ],
)
def get_prediction_result(
    schooling,
    hdi_income,
    hiv_aids,
    bmi,
    adult_mortality,
    alcohol,
    diphtheria,
    gdp
):
    selected_values = [
        schooling,
        hdi_income,
        hiv_aids,
        bmi,
        adult_mortality,
        alcohol,
        diphtheria,
        gdp
    ]
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
        #font_family="Helvetica",
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)