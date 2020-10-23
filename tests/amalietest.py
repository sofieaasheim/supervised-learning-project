import pandas as pd

# Import the entire data sets
math_df = pd.read_csv("../supervised-learning-project/data/student-mat.csv", sep=";")

# Divide into parameters and responses
parameter_df = math_df.drop(["G1", "G2", "G3"], axis=1)
response_df = math_df[["G1", "G2", "G3"]]