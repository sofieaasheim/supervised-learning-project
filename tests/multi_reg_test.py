import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import sklearn 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Import the entire data sets
df = pd.read_csv("./supervised-learning-project/data/life-expectancy.csv", sep=",") # OBS! må ha '../supervised-learning-project før /data på Emma sin

# Remove non-relevant parameters for the regression and remove all non-finite values such as NaN and +/- infinity
df.drop(['Country', 'Year', 'Status'], axis=1, inplace=True)
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
    y = df_regr['LifeExpectancy']
    X = sm.add_constant(X)
    # Ikke 100% sikker, men slik jeg forstår det er det her man skal dele datasettet in i train og test.
    # Dette er fordi når du lager regression model så trener du den samtidig. 
    # Dette resulterer i en litt dårligere modell pga. mindre datasett (litt lavere R²).
    # Men igjen så har man da et test dataset som man kan bruke til å predicte. 

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
    
    # Execute multiple regression using statsmodels
    regression_model = sm.OLS(y_train, X_train).fit()

    # Predicting using the model
    y_pred = regression_model.predict(X_test)
    df = pd.DataFrame({'Actual Life Expectanct':y_test, 'Predicted Life Excpectancy':y_pred})

    # Printing the STD between the predicted and the real Life Excpecanct
    # print("\n STD:")
    std = np.std(np.abs(y_test - y_pred))
    
    # Plotting the predicted and real values
    x = X_test
    y_t = np.round(y_test)
    y_p = np.round(y_pred)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x,y_t, c='b', marker='x', label='Actual Life Expectanct')
    ax1.scatter(x,y_p, c='r', marker='o', label='Predicted Life Excpectancy')
    plt.legend(loc='upper left')
    plt.show()


    return regression_model.summary(), df , 'STD:',  std, 

# Step 1: Multiple linear regression with AdultMortality, Alcohol, BMI, HIVAIDS, Income and Schooling
parameter_list_1 = ['AdultMortality', 'Alcohol', 'BMI', 'HIVAIDS', 'Income', 'Schooling']
#print(multiple_regression(df_regr, parameter_list_1))

# Step 2: Remove the parameter with the highest p-value. This was the Alchol parameter which had a
# p-value of 0.169. Do the regression again with the rest of the parameters
parameter_list_2 = ['AdultMortality', 'BMI', 'HIVAIDS', 'Income', 'Schooling']
print(multiple_regression(df_regr, parameter_list_2))

# Now all parameters have a p-value of less than 0.05, which indicates that all parameters
# are likely to be a meaningful addition to the model.


