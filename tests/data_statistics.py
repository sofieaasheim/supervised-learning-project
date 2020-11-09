import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Import the entire data sets
df = pd.read_csv("./supervised-learning-project/data/life-expectancy.csv", sep=",")

# Remove non-numerical parameters and the response
df.drop(['Country', 'Year', 'Status', 'LifeExpectancy'], axis=1, inplace=True)

# Make a parameter df and remove all non-finite values (NaN and +/- infinity)
parameter_df = df[np.isfinite(df).all(1)]

# Calculating the means:
print("The means:")
mean = df[['AdultMortality','InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling']].mean()
print(mean)

# Calculating the standard deviation
print("The standard deviatioon:")
std = df[['AdultMortality','InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling']].std() 
print(std)

#Calculationg the quantile
print("The quantile:")
quantile = df[['AdultMortality','InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling']].quantile()
print(quantile)