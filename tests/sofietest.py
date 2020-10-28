import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Import the entire data sets
math_df = pd.read_csv("../data/student-mat.csv", sep=";")

# Fjern dustekolonner
math_df.drop(['sex', 'address', 'school', 'Mjob', 'Fjob'], axis=1, inplace=True)

# Gjør om alle yes/no til 0/1
math_df.replace(('yes', 'no'), (1, 0), inplace=True)

# Fiks 'famsize' kolonnen
math_df.replace(('LE3', 'GT3'), (0, 1), inplace=True)

# Fiks 'Pstatus' kolonnen
math_df.replace(('A', 'T'), (0, 1), inplace=True)

# Fiks 'reason' kolonnen
math_df.replace(('home', 'reputation', 'course', 'other'), (1, 2, 3, 4), inplace=True)

# Fiks 'guardian' kolonnen
math_df.replace(('mother', 'father', 'other'), (1, 2, 3), inplace=True)

# Divide into parameters and responses
parameter_df = math_df.drop(["G1", "G2", "G3"], axis=1)
response_df = math_df[["G1", "G2", "G3"]]

print(math_df)



