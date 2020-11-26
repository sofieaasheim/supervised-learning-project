import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"

# Import the entire data set
df = pd.read_csv(data_url, sep=",")

# Remove non-relevant parameters for the regression and remove all non-finite values such as NaN and +/- infinity
df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]

# The best parameter combination from backward_selection.py
parameter_list = [
    "AdultMortality",
    "Alcohol",
    "GDP",
    "BMI",
    "HIVAIDS",
    "Diphtheria",
    "Income",
    "Schooling",
]

""" MAKING ERROR PLOTS FOR THE REGRESSION MODEL """


def error_plots(df_regr, parameter_list):
    # Make dataframe for parameters (X) and response (y)
    X = df_regr[parameter_list]
    y = df_regr["LifeExpectancy"]
    X = sm.add_constant(X)

    # Slpit the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Execute multiple regression using statsmodels on the training set
    regression_model = sm.OLS(y_train, X_train).fit()

    # Predict using the model and test parameters
    y_pred = regression_model.predict(X_test)

    # Compare the predicted value against the test value
    df = pd.DataFrame(
        {
            "Actual Life Expectancy": y_test,
            "Predicted Life Excpectancy": y_pred,
            "Error": (y_test - y_pred),
        }
    )

    # Calculate the error model
    error_model = np.sum((y_test - y_pred) ** 2)

    # Calculate the sum of average error
    error_sum = np.sum(y_test - y_pred)

    # Calculate the Mean Absolute Error
    mae = mean_absolute_error(y_pred, y_test)

    # Plot the predicted and real values against each other
    x = list(range(0, 330))
    y_t = y_test
    y_p = y_pred

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y_t,
            mode="markers",
            marker_symbol="x",
            marker_color="black",
            opacity=0.6,
            name="Actual Life Expectancy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_p,
            mode="markers",
            marker_color="crimson",
            name="Predicted Life Expectancy",
        )
    )
    fig.update_layout(
        yaxis_title="Years",
        xaxis_title="Different Instances",
        font_family="Helvetica",
        title="Actual vs. Predicted Life Expectancy",
        height=500,
        width=1000,
        showlegend=True,
    )
    fig.show()

    # Plot the error values
    fig = go.Figure(
        data=go.Scatter(
            x=list(range(0, 330)),
            y=y_test - y_pred,
            mode="markers",
            marker_symbol="x",
            marker_color="blue",
            opacity=0.6,
            name="Actual Life Expectancy",
        )
    )
    fig.update_layout(
        yaxis_title="Years",
        xaxis_title="Different Instances",
        font_family="Helvetica",
        title="Error in Actual vs. Predicted Life Expectancy",
        height=500,
        width=1000,
        showlegend=True,
    )
    fig.show()

    # Plot the absolute error values
    fig = go.Figure(
        data=go.Scatter(
            x=list(range(0, 330)),
            y=np.abs(y_test - y_pred),
            mode="markers",
            marker_symbol="x",
            marker_color="blue",
            opacity=0.6,
            name="Error in Prediction",
        )
    )
    fig.add_trace(go.Scatter(x=[0, 335], y=[mae, mae], name="Mean Absolute Error"))
    fig.update_layout(
        yaxis_title="Years",
        xaxis_title="Different Instances",
        font_family="Helvetica",
        title="Absolute Error in Actual vs. Predicted Life Expectancy",
        height=500,
        width=1000,
        showlegend=True,
    )
    fig.show()

    return (
        regression_model.summary(),  # The summary of the regression
        df,  # Table with actual and predicted life expecancy and the error between
        "ErrorModel:",
        error_model,  # Calculated ErrorModel
        "Mean Absolute Error:",
        mae,  # Calculated MAE
        "Sum of Errors:",
        error_sum,
    )  # Calculated Sum of Errors


print(error_plots(df_regr, parameter_list))
