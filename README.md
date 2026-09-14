# HW1 - Multiple Linear Regression (MLR)

## Problem 1: Create MLR Functions

Your goal is to write functions to:
- perform MLR using ordinary least squares
- perform an ANOVA test to assess the overall model usefulness
- perform coefficient hypothesis tests to assess coefficient usefulness

To do this, please complete the functions given in `mlr.py`. You are free to add any additional functions. Do not modify the `test_mlr.py` file.

Please code these functions from scratch (i.e., do not use any prebuilt OLS, ANOVA, or coefficient test functions, like `numpy.polyfit` or `scipy.optimize.least_squares`). You can use other helper functions though.

## Problem 2: Model and Analyze a Real Dataset

Your goal is to use MLR to model and analyze a real-world dataset: the Bureau of Transportation Statistics (BTS) Airline On-Time Performance Data (see [Resources](#resources) for access). You can define your own problem - for example, you could create a model to predict arrival flight delays and analyze whether various predictors are useful for such prediction.

To this, please update the `hw1.ipynb` Jupyter notebook file to:
- import the data you want to use
- produce a scatterplot matrix of the variables you plan to include in your model and describe your results
- fit an MLR model to your data
- analyze your model, including at least the following: (1) ANOVA model test, (2) individual coefficient hypothesis tests, and (3) an actual vs. predicted plot

I should be able to run your notebook to recreate your results. You can (and probably should) use existing MLR libraries for this task - I would suggest using the `statsmodels` library.

## Submission

1. Submit your updated `mlr.py` and `hw1.ipynb` files to your GitHub repository.
    1. You can do this through GitHub in a web browser by: click on your repository > click "Add file" > click Upload files > drag your updated files > click "Commit changes".
1. Submit your assignment on Gradescope by submitting your GitHub repository.

## Resources
Here are some resources that may be helpful:
- The BTS On-Time Performance dataset can be downloaded using this [link](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
    - The data can also be accessed by going to https://www.transtats.bts.gov > Data Finder > By Mode > Aviation > Airline On-Time Performance Data > Reporting Carrier On-Time Performance (1987-present) > Download
    - The dataset field name descriptions can be found using [link](https://www.transtats.bts.gov/TableInfo.asp?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr&V0s1_b0yB=D)
- The [statsmodel](https://www.statsmodels.org/stable/index.html) library
    - Note that the output from `statsmodel.ols.fit().summary()` is equivalent to the ANOVA and coefficient tests we discussed in class
- The `scipy.stats.f.sf` function (see [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html)) and `scipy.stats.t.sf` function (see [docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)) functions for getting p-values of a given test statistic
- The pandas [scatterplot matrix](https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html) function
- A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)
