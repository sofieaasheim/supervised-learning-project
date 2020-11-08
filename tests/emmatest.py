import pandas as pd
import numpy as np
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the entire data sets
dataset = pd.read_csv("../supervised-learning-project/data/life-expectancy.csv", sep=",")


#Fjern kolonner
dataset.drop(['AdultMortality', 'InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'Measles', 'BMI', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9', 'Income', 'Schooling'], axis=1, inplace=True)


#Creating initial dataframe
countrynames = dataset["Country"].unique()
country_df = pd.DataFrame(countrynames, columns=['Country'])

# converting type of columns to 'category'
#country_df['Country'] = country_df['Country'].astype('category')

# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
country_df['Country_Cat'] = labelencoder.fit_transform(country_df['Country'])
country_df

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')
# passing country-cat column (label encoded values of country)
enc_df = pd.DataFrame(enc.fit_transform(country_df[['Country_Cat']]).toarray())
# merge with  country_df  on key values
country_df = country_df.join(enc_df)
country_df
# merge with dataset on key values
dataset = dataset.merge(country_df)
dataset.drop(['Country','Year', 'Status'], axis=1, inplace=True)



X = dataset.iloc[:, 1:194].values
y = dataset.iloc[:, 0].values


#Splitting data into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
print(X_train.shape)
print(y_train.shape)
# regr = LinearRegression()
# regr.fit(X_train, y_train)

# print(regr.coef_)

# print(regr)

# Make predictions
# expected = y_test
# predicted = regr.predict(X_test)

# print(predicted)


# Summarize the fit of the model
#mse = np.mean((predicted-expected)**2)
# print (regr.intercept_, regr.coef_, mse) 
# print(regr.score(X_train, y_train))