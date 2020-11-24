import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split


""" MAKE THE MULTIPLE REGRESSION MODEL FOR PREDICTION """


def multiple_regression(df_regr, parameter_list):
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

    # Print the average error between the predicted value and the thest value
    error = np.mean(np.abs(y_test - y_pred))

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
    # Uncomment to show prediction vs. actual values plot
    # fig.show()

    return regression_model.summary(), df, "Average Error:", error
