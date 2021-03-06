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

data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"

# Import the entire data set
dataset = pd.read_csv(data_url, sep=",")

# Remove columns - OBS! Many different combinations of parameters have been tried.
dataset.drop(['Status', 'InfantDeaths', 'Alcohol', 'PercentageExpenditure', 'HepatitisB', 'BMI', 'Measles', 'UnderFiveDeaths', 'Polio', 'TotalExpenditure', 'Diphtheria', 'HIVAIDS', 'GDP', 'Population', 'Thinness1_19', 'Thinness5_9'], axis=1, inplace=True)

# Create the initial dataframe
countrynames = dataset["Country"].unique()
country_df = pd.DataFrame(countrynames, columns=['Country'])
# # converting type of columns to 'category'
# #country_df['Country'] = country_df['Country'].astype('category')

# Create instance of labelencoder
labelencoder = LabelEncoder()

# Assign numerical values and store in another column
country_df['Country_Cat'] = labelencoder.fit_transform(country_df['Country'])
country_df

# Create instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# Pass country-cat column (label encoded values of country)
enc_df1 = pd.DataFrame(enc.fit_transform(country_df[['Country_Cat']]).toarray())

# Merge with country_df on key values
country_df = country_df.join(enc_df1)
country_df

# Merge with dataset on key values
dataset = dataset.merge(country_df)
dataset.drop(['Country','Year'], axis=1, inplace=True)
dataset.dropna(inplace=True)

# Divide into response and parameters
X = dataset.iloc[:,1:].values
X = sm.add_constant(X) #Add constant
y = dataset.iloc[:, 0].values 

# Split the data into test and training data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# Build model OBS! Two different methods have been attempted, that is why some code is commented out
#regr = LinearRegression()
regr = sm.OLS(y_train,X_train).fit()

#regr.fit(X_train, y_train)

# Print summary 
print(regr.summary())

#Make predictions
#expected = y_test
#predicted = regr.predict(X_test)

#print(expected)
#print(predicted)


# # Summarize the fit of the model
#mse = np.mean((predicted-expected)**2)
#print (regr.intercept_, regr.coef_, mse) 
# print(regr.score(X_train, y_train)