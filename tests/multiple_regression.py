import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

""" IMPORT AND PREPROCESS THE DATA """

data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"

# Import the entire data sets
df = pd.read_csv(data_url, sep=",")

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

""" MAKING THE MULTIPLE REGRESSION PREDICTION MODEL """

# Step 1: Remove non-linear parameters and check for multicollinearlity
# The first thing we need to to is removing the parameters that do not seem to show any linear 
# relationship with the response by using the correlation plots from above (these are also
# visualized on our website tdt4173group9.herokuapp.com). These are:

initial_parameters = ['AdultMortality', 'InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'BMI', 'Polio',
'TotalExpenditure', 'HIVAIDS', 'Diphtheria', 'Thinness1_19', 'Income', 'Schooling']

# UnderFiveDeaths, GDP and Thinness 5_9 also seemed to have a slight linear relationship with the 
# response, but these parameters are highly correlated with InfantDeaths, PercentageExpenditure 
# and Thinness1_19respectively, and cannot be in the same model simultaneously.
# The correlations between all parameters are visualized in correlation_matrix.py

# Step 2: Make a multiple regression model for testing and training of the data, and prediction

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
    # Uncomment to show prediction vs actual values plot
    #fig.show()

    return regression_model.summary(), df, "Average Error:", error


# Step 3: Execute multiple linear regression with all the initial parameters described above

#print(multiple_regression(df_regr, initial_parameters))

# Step 4: Remove the parameter with the highest p-value. This was the Thinness1_19 parameter which had a
# p-value of 0.481. Do the regression again with the rest of the parameters.

test1_parameters = initial_parameters
test1_parameters.remove("Thinness1_19")
#print(multiple_regression(df_regr, test1_parameters))

# Step 5: Repeat step 4 until all parameters have p-values of less than 0.05.

# The next highest p-value parameter is HepatitisB
test2_parameters = test1_parameters
test2_parameters.remove("HepatitisB")
#print(multiple_regression(df_regr, test2_parameters))

# The next highest p-value parameter is Polio
test3_parameters = test2_parameters
test3_parameters.remove("Polio")
#print(multiple_regression(df_regr, test3_parameters))

# The next highest p-value parameter is TotalExpenditure
test4_parameters = test3_parameters
test4_parameters.remove("TotalExpenditure")
#print(multiple_regression(df_regr, test4_parameters))

# The next highest p-value parameter is InfantDeaths
test5_parameters = test4_parameters
test5_parameters.remove("InfantDeaths")
print(multiple_regression(df_regr, test5_parameters))

# Now all parameters have a p-value of less than 0.05, which indicates that all parameters
# are likely to be a meaningful addition to the model. The parameters left are:

# AdultMortality, Achohol, PercentageExpenditure, BMI, HIVAIDS, Diptheria, Income, Schooling

# Step 6: Try exchanging multicollinear parameter
# We can also try to exhange PercentageExpenditure with GDP to see if the model improves:

test6_parameters = test5_parameters
test6_parameters.remove("PercentageExpenditure")
test6_parameters.append("GDP")
print(multiple_regression(df_regr, test6_parameters))

# The R-squared and adjusted R-squared are so close in values with these two different
# parameter combinations, so it does (almost) not make any difference
