import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Import the entire data sets
df = pd.read_csv("./supervised-learning-project/data/life-expectancy.csv", sep=",")

# Remove non-numerical parameters 
df.drop(['Country', 'Year', 'Status'], axis=1, inplace=True)

# Make a parameter df and remove all non-finite values (NaN and +/- infinity)
parameter_df = df[np.isfinite(df).all(1)]

# Utilizing the tools within pandas for all the calculations
# Calculating the means:
print("The means:")
mean = df[['LifeExpectancy','AdultMortality','InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling']].mean()
print(mean)

# Calculating the standard deviation
print("The standard deviatioon:")
std = df[['LifeExpectancy', 'AdultMortality','InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling']].std() 
print(std)

#Calculationg the quantile
print("The quantile:")
quantile = df[['LifeExpectancy', 'AdultMortality','InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling']].quantile()
print(quantile)

# Plotting using matplotlib
x = 'LifeExpectancy'  
y = 'AdultMortality'

df.plot.scatter(x, y)
# plt.show() 
#boxplot = df.boxplot(by = y, column = [x], grid = False)
boxplot = df.boxplot(by = x, column = [y], grid = False)
plt.show()