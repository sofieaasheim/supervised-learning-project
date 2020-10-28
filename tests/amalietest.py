import pandas as pd

# Import the entire data sets
math_df = pd.read_csv("./supervised-learning-project/data/student-mat.csv", sep=";")

# Divide into parameters and responses
parameter_df = math_df.drop(["G1", "G2", "G3"], axis=1)
response_df = math_df[["G1", "G2", "G3"]]

# Boolean variables
math_df.replace(('yes', 'no'), (1, 0), inplace=True)

# Dummie variables, test with sex column 
sex_df = math_df[["sex"]]
dummies = pd.get_dummies(sex_df)

# print(math_df)
print(dummies)