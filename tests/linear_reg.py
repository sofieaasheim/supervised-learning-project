#Import Library
import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

# Loading data 
data = pd.read_csv("./supervised-learning-project/data/student-mat.csv", sep=";")

# Trimming data 
data = data[["freetime", "age", "health", "Dalc", "Walc", "Medu", "Fedu", "G3"]]
data = shuffle(data) # Optional - shuffle the data

# Separating data 
predict = "G3" # = respons

x = np.array(data.drop([predict], 1)) # parameters
y = np.array(data[predict]) # respons

# Splitting in testing and training sets 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(20):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # Implementing linear regression 
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f: # Saving the model if it has a better score than one we've already trained
            pickle.dump(linear, f)

print("Best accuracy:")
print(best)


# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_) # each slope value
print('Intercept: \n', linear.intercept_) 
print("-------------------------")

# List of all predictions
print("List of predictions:")
predicted = linear.predict(x_test)
#predicted_data = list[]
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])
    #predicted_data.append(predicted[x])
#printe

# Drawing and plotting model
plot = "Fedu"
plt.scatter(data[plot], data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()


# Plotting predicted grade against grade

#plt.scatter(predicted)