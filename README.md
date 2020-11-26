# Supervised Learning Project

This repository contains a machine learning project using a multiple regression model to predict life ecpectancy. These YouTube videos are a nice place to start for understanding the method:

- [The Very Basics]
- [Data Preparation]


A report describing the theory behind the project, the data set, the project process, and the results can be found in the blabla.pdf file.

Some of the project results are presented in an interactive website [here], where you can predict the life excpectancy based on parameters that you decide the value of. This website also breifly describes the theory behind the methods used. 


## Data Set

The data set used in the project is from the World Health Organization (WHO). It consists of the life expectancy for 193 countries over a period of 15 years, and different health factors, affecting the life expectancy. The life excpectancy is the response while the health factors are the parameters. The data set can be found here: [WHO data set].

Relevant data for the project can be found in the `data` folder:

- `life-expectancy.csv` contains the relevant data set for the project.

- `life-expectancy.txt` contains information about all the variables in the data set.


## The Code

You will find the code for the model and the experiments used in the project in the folder `tests`. Here there are six files: 

- `multiple_regression.py`: The multiple regression model is made in the function `multiple_regression(df_regr, parameter_list)`. This function takes a data frame and a list of parameter names as arguments. First, it splits the data in training and testing sets, then performs a multiple regression as well as  plotting the predicted life expectancy and the responses. This function is imported to and used in some of the other files in the project.

- `backward_elimination.py`: Uses the backward elimination method to identify the most important parameters so that the model is reliable. This is done in four steps, which are explained as comments in the file. 

- `forward_selection.py`: Experiments with the forward selection method.

- `correlation_matrix.py`: Makes a correlation matrix showing the correlation coeffictients of all the parameter pairs and visualizes it in a heatmap.

- `error_plots.py`: Evaluates the model by calculationg the errors, sum of errors, mean average error and plotting these. 

- `dummy_variables.py`: Experiments with dummy variables on the Country parameter.

The results of the project is, as mentioned, visualized on an interactive website. The code for this website is in the `app.py` file in the main folder.


## Technologies
In this project, the following technologies are used:

- **Python** – The programming language used in the project
- **Dash** – A framework for building data visualization web applications
- **Pandas** – A data analysis Python library
- [scikit-learn] and [statsmodels] – Tools such as classes and functions for predictive data analysis (useful for the multiple regression analysis)


## Installation

To contribute to the project, first make a project directory locally on your computer. Inside this, create a [virtual environment] for the project. In this project directory, you should also clone this repository to a new folder. Some of the packages that are necessary to install for the project to run locally:

```sh
pip install pandas
pip install dash
pip install statsmodels
pip install -U scikit-learn
```

The rest of the packages needed for the project can be found in the `requirements.txt` file.


## Usage

- The website is made in the `app.py` file.
- For experimentation and testing, create `.py` files in the `tests` folder.

To import the enitre data set to a testing file in the `tests` folder, i.e. use the following:

```sh
import pandas as pd

data_url = "https://raw.githubusercontent.com/sofieaasheim/supervised-learning-project/main/data/life-expectancy.csv"
df = pd.read_csv(data_url, sep=",")
```

When performing tests, make sure that you only push to the **tests** branch (or make a new branch). The **main** branch should only be used for pushing finished stuff.

To run the application locally, type `python app.py` in the terminal while in the `supervised-learning-project` folder and go to http://127.0.0.1:8050/ in your browser.

To run the test files, type `python name_of_file.py` in the terminal while in the `tests` folder.


## Deployment

The code for the website is deployed directly from the **main** branch through Heroku. The web page can be found at https://tdt4173group9.herokuapp.com/.



[The Very Basics]: <https://www.youtube.com/watch?v=dQNpSa-bq4M>
[Data Preparation]: <https://www.youtube.com/watch?v=2I_AYIECCOQ&list=TLPQMTkxMDIwMjCcYgA12J8jGg&index=2>
[10 minutes to pandas]: <https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html#min>
[Dash tutorial]: <http://dash.plotly.com/installation>
[scikit-learn]: <https://scikit-learn.org/stable/>
[statsmodels]: <https://www.statsmodels.org/stable/index.html>
[virtual environment]: <https://www.geeksforgeeks.org/python-virtual-environment/>
[GitHub Desktop]: <https://desktop.github.com/>
[WHO data set]: <https://www.kaggle.com/kumarajarshi/life-expectancy-who?fbclid=IwAR1NONmZtX8ZlR_I3sZBL04069sSHin8VPVsoN3lJehHfnBK0eKXpbEz3-U>
[here]: <https://tdt4173group9.herokuapp.com/>
