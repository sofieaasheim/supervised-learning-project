import pandas as pd

d1 = pd.read_csv("./supervised-learning-project/data/student-mat.csv", sep=";")
d2 = pd.read_csv("./supervised-learning-project/data/student-por.csv", sep=";")

d3 = pd.merge(d1,d2,on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
print(d3) # 382 students