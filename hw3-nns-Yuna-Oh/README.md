# HW3 - Neural Networks (NNs)

## Problem 1: Go through a PyTorch Tutorial

Your goal is to learn how to use PyTorch to implement and train neural networks.

To do this, please go through the [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) tutorial. Please update the `hw3_pytorch.ipynb` Jupyter notebook file as you go through the tutorial.

You can also refer to the [Introduction to PyTorch - YouTube Series](https://docs.pytorch.org/tutorials/beginner/introyt/introyt_index.html) tutorial series for additional information as needed.

## Problem 2: Model and Analyze a Real Dataset

Your goal is to use a neural network to model and analyze a real-world dataset: the Bureau of Transportation Statistics (BTS) Airline On-Time Performance Data (see [Resources](#resources) for access). You can define your own problem - for example, you could create a model to predict whether or not an arrival flight will be delayed.

You can formulate your problem as a regression or classification problem.

To this, please update the `hw3_bts.ipynb` Jupyter notebook file to:
- define your problem and import the data you want to use
- train a neural network using your data (training data)
- also fit **at least one** of the following models: multiple linear regression, KNN, logistic regression, LDA, or QDA
- analyze your models and discuss your results, including at least the following: (1) a representation of a training curve (e.g., loss vs. epochs) and (2) a representation of test error

I should be able to run your notebook to recreate your results. Please use `pytorch` to train your neural network.

## Submission

1. Submit your updated `hw3_pytorch.ipynb` and `hw3_bts.ipynb`files to your GitHub repository.
    1. You can do this through GitHub in a web browser by: click on your repository > click "Add file" > click Upload files > drag your updated files > click "Commit changes".
1. Submit your assignment on Gradescope by submitting your GitHub repository.

## Resources
Here are some resources that may be helpful:
- The BTS On-Time Performance dataset can be downloaded using this [link](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
    - The data can also be accessed by going to https://www.transtats.bts.gov > Data Finder > By Mode > Aviation > Airline On-Time Performance Data > Reporting Carrier On-Time Performance (1987-present) > Download
    - The dataset field name descriptions can be found using [link](https://www.transtats.bts.gov/TableInfo.asp?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr&V0s1_b0yB=D)
- A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)
