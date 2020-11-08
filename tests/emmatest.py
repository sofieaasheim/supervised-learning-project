import pandas as pd
import numpy as np
#from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import the entire data sets
dataset = pd.read_csv("../supervised-learning-project/data/student-mat.csv", sep=";")
#math_df = pd.read_csv("../data/student-mat.csv", sep=";")


#Fjern kolonner
dataset.drop(['sex', 'address', 'school', 'Mjob', 'Fjob'], axis=1, inplace=True)


# Gjør om alle yes/no til 0/1
dataset.replace(('yes', 'no'), (1, 0), inplace=True)

# Fiks 'famsize' kolonnen
dataset.replace(('LE3', 'GT3'), (0, 1), inplace=True)

# Fiks 'Pstatus' kolonnen
dataset.replace(('A', 'T'), (0, 1), inplace=True)

# Fiks 'reason' kolonnen
dataset.replace(('home', 'reputation', 'course', 'other'), (1, 2, 3, 4), inplace=True)

# Fiks 'guardian' kolonnen
dataset.replace(('mother', 'father', 'other'), (1, 2, 3), inplace=True)

# Divide into parameters and responses
X = dataset.drop(["G1", "G2", "G3"], axis=1)
y = dataset[["G1", "G2", "G3"]]

#print(dataset)
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, 27].values

# labelencoder = LabelEncoder()
# X[:, 2] =labelencoder.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categories= 2)
# X = onehotencoder.fit_transform(X).toarray()

# print(dataset)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

regr = LinearRegression()
regr.fit(X_train, y_train)

# print(regr.coef_)

# print(regr)

# Make predictions
expected = y_test
predicted = regr.predict(X_test)

print(predicted)


# Summarize the fit of the model
mse = np.mean((predicted-expected)**2)
# print (regr.intercept_, regr.coef_, mse) 
# print(regr.score(X_train, y_train))