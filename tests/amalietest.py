import pandas as pd

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

math_df = pd.concat([math_df, dummies], axis=1)
math_df = math_df.drop(["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"], axis=1)

print(math_df)