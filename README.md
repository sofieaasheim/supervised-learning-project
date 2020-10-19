# Supervised learning project

This is a project using multiple regression from supervised learning to obtain information about student performance. These YouTube videos are a nice place to start for understanding the method:
- [The Very Basics]
- [Data Preparation]


# Technologies
In this project, we will use several technologies:

- Python – The programming language used in the project
- Dash – A framework for building data visualisation web applications
- Pandas – A data analysis Python library

If you are unfamiliar with these, I reccommend doing the [Dash tutorial] and looking at the [10 minutes to pandas] short user guide.


# Data sets

Relevant data sets for the project can be found in the `data` folder.
- `student-mat.csv` contains data from students in a math course
- `student-por.csv` contains data from students in a Portugese language course


Row 1-30 in the data sets are parameters. Row 31-33 in the data sets are grades (responses). 

When working with the data, be aware thate the responses are the dependent variables and the parameters are the independent variables. They should therefore be separated into two different data frames.

More information about the data can be found in the `student.txt` file.


# Usage

- The finished product will be in the `app.py` file. Do not edit this before having useful results. 
- For experimentation and testing, create `.py` files in the `tests` folder.

To import the data sets to the testing files in the `tests` folder, i.e. use the following:

```sh
import pandas as pd

math_df = pd.read_csv("./supervised-learning-project/data/student-mat.csv", sep=";")
portugese_df = pd.read_csv("./supervised-learning-project/data/student-por.csv", sep=";")
```

When performing tests, make sure that you only push to the "tests" branch, or make another branch. The "main" branch should only be used for pushing finished stuff.


# Deployment

The web page are automatically deployed when pushing to the "main" branch. This can be found at https://tdt4173group9.herokuapp.com/.



[The Very Basics]: <https://www.youtube.com/watch?v=dQNpSa-bq4M>
[Data Preparation]: <https://www.youtube.com/watch?v=2I_AYIECCOQ&list=TLPQMTkxMDIwMjCcYgA12J8jGg&index=2>
[10 minutes to pandas]: <https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min>
[Dash tutorial]: <http://dash.plotly.com/installation>
