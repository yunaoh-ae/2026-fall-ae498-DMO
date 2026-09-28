# HW2 - Classification

## Problem 1: Create Classification Functions

Your goal is to write a function to:
- fit a KNN classifier

To do this, please complete the functions given in `classification.py`. You are free to add any additional functions. Do not modify the `test_classification.py` file.

## Problem 2: Model and Analyze a Real Dataset

Your goal is to use a classifier to model and analyze a real-world dataset: the Bureau of Transportation Statistics (BTS) Airline On-Time Performance Data (see [Resources](#resources) for access). You can define your own problem - for example, you could create a model to predict whether or not an arrival flight will be delayed.

To this, please update the `hw2.ipynb` Jupyter notebook file to:
- import the data you want to use
- fit a KNN classifer to your data
- also fit **at least one** of the following classifiers to your data: logistic regression, LDA, or QDA
- analyze your models, including at least the following: (1) test error rate and (2) AUC
- analyze how the test mean squared error of the KNN model changes as a function of K

I should be able to run your notebook to recreate your results. You can (and probably should) use existing libraries for this task, such as the `scikit-learn` library.

## Problem 3 (4 credit hours): LDA vs. QDA Concept Questions

Show why choosing the class $k$ that maximizes the discriminant function is equivalent to choosing the class $k$ that maximizes the conditional distribution $Pr(Y = k | X = x)$.

Show why removing the assumption of constant class variance for the Gaussian distribution of $f_k(x)$ results in a discriminant function that is quadratic in $x$.

You can assume $p=1$ for this problem and submit your solution in the `hw2.ipynb` file or a separate scanned `hw2.pdf` file.

## Submission

1. Submit your updated `classification.py` and `hw2.ipynb` files to your GitHub repository.
    1. You can do this through GitHub in a web browser by: click on your repository > click "Add file" > click Upload files > drag your updated files > click "Commit changes".
1. Submit your assignment on Gradescope by submitting your GitHub repository.

## Resources
Here are some resources that may be helpful:
- The BTS On-Time Performance dataset can be downloaded using this [link](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
    - The data can also be accessed by going to https://www.transtats.bts.gov > Data Finder > By Mode > Aviation > Airline On-Time Performance Data > Reporting Carrier On-Time Performance (1987-present) > Download
    - The dataset field name descriptions can be found using [link](https://www.transtats.bts.gov/TableInfo.asp?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr&V0s1_b0yB=D)
- The [scikit-learn](https://scikit-learn.org/stable/) library
- A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)
