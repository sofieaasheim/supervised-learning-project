# Supervised learning project

This is a project using multiple regression from supervised learning to obtain information about student performance. These YouTube videos are a nice place to start for understanding the method:
- [The Very Basics]
- [Data Preparation]

## Installation

To contribute to the project, first make a project directory locally on your computer. Inside this, create a [virtual environment] for the project. In this project directory, you should also clone this repository. Some packages that are necessary for the project:

```sh
pip install pandas
pip install dash
pip install statsmodels
pip install -U scikit-learn
```

I recommend downloading and using [GitHub Desktop] when working with git and the project to make things a bit easier for yourself.

## Technologies
In this project, we will use several technologies:

- **Python** – The programming language used in the project
- **Dash** – A framework for building data visualisation web applications
- **Pandas** – A data analysis Python library
- [scikit-learn] and [statsmodels] – Tools such as classes and functions for predictive data analysis (useful for the multiple regression analysis)

If you are unfamiliar with these, I reccommend doing the [Dash tutorial] and looking at the [10 minutes to pandas] short user guide.


## Data sets

Relevant data sets for the project can be found in the `data` folder.
- `student-mat.csv` contains data from students in a math course
- `student-por.csv` contains data from students in a Portugese language course


Row 1-30 in the data sets are parameters. Row 31-33 in the data sets are grades (responses). 

When working with the data, be aware thate the responses are the dependent variables and the parameters are the independent variables. They should therefore be separated into two different data frames.

More information about the data can be found in the `student.txt` file.


## Usage

- The finished product will be in the `app.py` file. If you edit this, make sure to push to the **tests** branch before pushing to **main**.
- For experimentation and testing, create `.py` files in the `tests` folder.

To import the data sets to the testing files in the `tests` folder, i.e. use the following:

```sh
import pandas as pd

math_df = pd.read_csv("../supervised-learning-project/data/student-mat.csv", sep=";")
portugese_df = pd.read_csv("../supervised-learning-project/data/student-por.csv", sep=";")
```

When performing tests, make sure that you only push to the **tests** branch (or make a new branch). The **main** branch should only be used for pushing finished stuff.

To run the application locally, type `python app.py` in the terminal while in the **supervised-learning-project** folder and go to http://127.0.0.1:8050/ in your browser.


## Deployment

The code is automatically deployed when pushing to the **main** branch. The web page can be found at https://tdt4173group9.herokuapp.com/.



[The Very Basics]: <https://www.youtube.com/watch?v=dQNpSa-bq4M>
[Data Preparation]: <https://www.youtube.com/watch?v=2I_AYIECCOQ&list=TLPQMTkxMDIwMjCcYgA12J8jGg&index=2>
[10 minutes to pandas]: <https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min>
[Dash tutorial]: <http://dash.plotly.com/installation>
[scikit-learn]: <https://scikit-learn.org/stable/>
[statsmodels]: <https://www.statsmodels.org/stable/index.html>
[virtual environment]: <https://www.geeksforgeeks.org/python-virtual-environment/>
[GitHub Desktop]: <https://desktop.github.com/>
