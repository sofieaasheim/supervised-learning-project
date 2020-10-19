import pandas as pd

math_df = pd.read_csv("../supervised-learning-project/data/student-mat.csv", sep=";")
portugese_df = pd.read_csv("../supervised-learning-project/data/student-por.csv", sep=";")

merged_df = pd.merge(math_df,portugese_df,on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
print(merged_df) # 382 students