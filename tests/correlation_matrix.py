import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import the entire data set
df = pd.read_csv("../data/life-expectancy.csv", sep=",")

# Remove non-numerical parameters and the response
df.drop(["Country", "Year", "Status", "LifeExpectancy"], axis=1, inplace=True)

# Make a parameter df and remove all non-finite values (NaN and +/- infinity)
parameter_df = df[np.isfinite(df).all(1)]

# Array for all correlation coefficients using the built-in .corr() method from pandas using
# Pearson standard correlation coefficient
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
correlation_df = parameter_df.corr()

# Plot the correlation coefficients in a heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=correlation_df,
        x=list(correlation_df.columns),
        y=list(correlation_df.columns),
        hoverongaps=False,
    )
)
fig.update_layout(
    title="Correlation matrix",
    font_family="Helvetica",
)
fig.show()
