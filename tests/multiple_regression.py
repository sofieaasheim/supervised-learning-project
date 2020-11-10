import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

# Import the entire data sets
df = pd.read_csv("../data/life-expectancy.csv", sep=",") # OBS! må ha '../supervised-learning-project før /data på Emma sin

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
    
    # Execute multiple regression using statsmodels
    regression_model = sm.OLS(y, X).fit()

    return regression_model.summary()

# Step 1: Multiple linear regression with AdultMortality, Alcohol, BMI, HIVAIDS, Income and Schooling
parameter_list_1 = ['AdultMortality', 'Alcohol', 'BMI', 'HIVAIDS', 'Income', 'Schooling']
#print(multiple_regression(df_regr, parameter_list_1))

# Step 2: Remove the parameter with the highest p-value. This was the Alchol parameter which had a
# p-value of 0.169. Do the regression again with the rest of the parameters
parameter_list_2 = ['AdultMortality', 'BMI', 'HIVAIDS', 'Income', 'Schooling']
print(multiple_regression(df_regr, parameter_list_2))

# Now all parameters have a p-value of less than 0.05, which indicates that all parameters
# are likely to be a meaningful addition to the model.

# Step 3: Training and testing the model ??? Er dette riktig rekkefølge? Kan bytte til
# å bruke sklearn nå hvis det er lettere evt, tror det er mer tilpasset training/test