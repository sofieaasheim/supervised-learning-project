import pandas as pd
import numpy as np
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Import the entire data sets
dataset = pd.read_csv("../supervised-learning-project/data/life-expectancy.csv", sep=",")

#Fjern kolonner
dataset.drop(['Status', 'InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'BMI', 'Measles', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9'], axis=1, inplace=True)

# Fiks 'status' kolonnen
#dataset.replace(('Developing', 'Developed'), (0, 1), inplace=True)

#Creating initial dataframe
countrynames = dataset["Country"].unique()
country_df = pd.DataFrame(countrynames, columns=['Country'])
# # converting type of columns to 'category'
# #country_df['Country'] = country_df['Country'].astype('category')

# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
country_df['Country_Cat'] = labelencoder.fit_transform(country_df['Country'])
country_df

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing country-cat column (label encoded values of country)
enc_df1 = pd.DataFrame(enc.fit_transform(country_df[['Country_Cat']]).toarray())
# merge with  country_df  on key values
country_df = country_df.join(enc_df1)
country_df
# merge with dataset on key values
dataset = dataset.merge(country_df)
dataset.drop(['Country','Year'], axis=1, inplace=True)
dataset.dropna(inplace=True)

print(dataset)
X = dataset.iloc[:,1:].values
X = sm.add_constant(X)
y = dataset.iloc[:, 0].values

print(y)



#Splitting data into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
regr = LinearRegression()

regr.fit(X_train, y_train)


#print(regr)

#Make predictions
expected = y_test
predicted = regr.predict(X_test)

print(expected)
print(predicted)


# Summarize the fit of the model
mse = np.mean((predicted-expected)**2)
print (regr.intercept_, regr.coef_, mse) 
print(regr.score(X_train, y_train))

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

regression_results(expected, predicted)
