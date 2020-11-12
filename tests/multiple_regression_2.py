import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import the entire data sets
df = pd.read_csv(
    "./supervised-learning-project/data/life-expectancy.csv", sep=","
)  # OBS! må ha '../supervised-learning-project før /data på Emma sin

# Remove non-relevant parameters for the regression and remove all non-finite values such as NaN and +/- infinity
df.drop(["Country", "Year", "Status"], axis=1, inplace=True)
df_regr = df[np.isfinite(df).all(1)]

# Make correlation plots for all parameters to find linear relationships
"""
for col in X:
    fig = go.Figure(data=go.Scatter(x=X[col], y=y, mode='markers'))
    fig.update_layout(title=col)
    fig.show()
"""

""" MULTIPLE REGRESSION
Now, perform a multiple regression on the parameters that show a linear relationship with the response,
using the plots from above. These parameters are:

AdultMortality, AlcoholConsumption, BMI, HIVAIDS, Income and Schooling
"""


def multiple_regression(df_regr, parameter_list):
    # Make dataframe for parameters (X) and response (y)
    X = df_regr[parameter_list]
    y = df_regr["LifeExpectancy"]
    X = sm.add_constant(X)

    # Slpitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Execute multiple regression using statsmodels on the training set
    regression_model = sm.OLS(y_train, X_train).fit()

    # Predicting using the model and test parameters
    y_pred = regression_model.predict(X_test)

    # Comparing the predicted value against the test value
    df = pd.DataFrame(
        {
            "Actual Life Expectancy": y_test,
            "Predicted Life Excpectancy": y_pred,
            "Error": (y_test - y_pred),
        }
    )

    # Printing the average error between the predicted value and the thest value
    error = np.mean(np.abs(y_test - y_pred))

    # Plotting the predicted and real values against each other
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
            name="Actual life expectancy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_p,
            mode="markers",
            marker_color="crimson",
            name="Predicted life expectancy",
        )
    )
    fig.update_layout(
        yaxis_title="Years",
        font_family="Helvetica",
        title="Actual vs. predicted life expectancy",
        height=500,
        width=1000,
    )
    fig.show()

    return regression_model.summary(), df, "Average Error:", error


# Step 1: Multiple linear regression with AdultMortality, Alcohol, BMI, HIVAIDS, Income and Schooling
parameter_list_1 = [
    "AdultMortality",
    "Alcohol",
    "BMI",
    "HIVAIDS",
    "Income",
    "Schooling",
]
# print(multiple_regression(df_regr, parameter_list_1))

# Step 2: Remove the parameter with the highest p-value. This was the Alchol parameter which had a
# p-value of 0.169. Do the regression again with the rest of the parameters
parameter_list_2 = ["AdultMortality", "BMI", "HIVAIDS", "Income", "Schooling"]
# print(multiple_regression(df_regr, parameter_list_2))

# Now all parameters have a p-value of less than 0.05, which indicates that all parameters
# are likely to be a meaningful addition to the model.
