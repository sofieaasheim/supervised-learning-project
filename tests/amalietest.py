import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Import the entire data sets
math_df = pd.read_csv("./supervised-learning-project/data/student-mat.csv", sep=";")

# Divide into parameters and responses
parameter_df = math_df.drop(["G1", "G2", "G3"], axis=1)
response_df = math_df[["G1", "G2", "G3"]]

# Boolean variables
math_df.replace(('yes', 'no'), (1, 0), inplace=True)

# Dummie variables, test with all text columbs
dummies_df = math_df[["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]]
dummies = pd.get_dummies(dummies_df)
print(dummies)

# Exchanging string variables with dimmy variables
math_df = pd.concat([math_df, dummies], axis=1)
math_df = math_df.drop(["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"], axis=1)

# Splitting into training and test sets 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(parameter_df, response_df, test_size = 0.1)

# Defining type of regression
linear = linear_model.LinearRegression()

# Accuracy of the model 
#linear.fit(x_train, y_train)
#acc = linear.score(x_test, y_test)
#print(acc)

#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

print(math_df)