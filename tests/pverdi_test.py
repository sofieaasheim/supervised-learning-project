import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Import entire dataset
dataset = pd.read_csv("../supervised-learning-project/data/life-expectancy.csv", sep=",")

# Remove parameters
dataset.drop([ 'Year', 'Country','Status', 'InfantDeaths', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9'], axis=1, inplace=True)

# Check for NaN 
# print(np.any(np.isnan(dataset))) #and gets False
# print(np.all(np.isfinite(dataset)))

# Remove rows with NaN
dataset.dropna(inplace=True)

# Dependent and independent variables
X = dataset[['Schooling', 'Income', 'AdultMortality', 'BMI', 'HIVAIDS']]
y = dataset['LifeExpectancy']
X = sm.add_constant(X)
# Split dataset into test data and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(y_test)
# Fit model
model = sm.OLS(y_train,X_train).fit()

print(model.predict(X_test))

print(model.summary())