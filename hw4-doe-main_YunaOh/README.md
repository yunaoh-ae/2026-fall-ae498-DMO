# HW4 - Design of Experiments

## Problem 1: Create Main Effects and Interactions Functions

Your goal is to write functions to:
- calculate main and interaction effects from a $2^2$ factorial experiment with $n$ replications
- create a line plot of effects from a $2^2$ factorial experiment with $n$ replications
- create a Tornado plot of effects from a $2^2$ factorial experiment with $n$ replications

To do this, please complete the functions given in `doe.py`. You are free to add any additional functions. Do not modify the `test_doe.py` file.

See `effects-example.png` and `tornado-example.png` for example plots.

## Problem 2: Model and Analyze a Dataset

Your goal is to use your functions to model and analyze the `circuit-board.csv` dataset.

A router is used to cut locating notches on a printed circuit board. The vibration level at the surface of the board as it is cut is considered to be a major source of dimensional variation in the notches. Two factors are thought to influence vibration: bit size (A) and cutting speed (B). Two bit sizes (1/16 and 1/8 in.) and two speeds (40 and 90 rpm) are selected, and four boards are cut at each set of conditions shown below. The response variable is vibration measured as the resultant vector of three accelerometers (x, y, and z) on each test circuit board.

| A | B | Replicate | | | |
| - | - | ---- | ---- | ---- | ---- |
|   |   | I    | II   | III  | IV   |
| - | - | 18.2 | 18.9 | 12.9 | 14.4 |
| + | - | 27.2 | 24.0 | 22.4 | 22.5 |
| - | + | 15.9 | 14.5 | 15.1 | 14.2 |
| + | + | 41.0 | 43.9 | 36.3 | 39.9 |

To this, please update the `hw4.ipynb` Jupyter notebook file to:
- define your problem and import the data you want to use
- analyze and discuss the main and interaction effects using your effects function
- analyze and discuss the data using ANOVA (you can use your own code or a pre-built library)
- discuss what levels of bit size and speed you would recommend for routine operation and why?

I should be able to run your notebook to recreate your results. You can (and probably should) use existing libraries for this task, aside from calculating main and interaction effects.

## Submission

1. Submit your updated `doe.py` and `hw4.ipynb` files to your GitHub repository.
    1. You can do this through GitHub in a web browser by: click on your repository > click "Add file" > click Upload files > drag your updated files > click "Commit changes".
1. Submit your assignment on Gradescope by submitting your GitHub repository.

## Notes
- `catalyst-reactant.csv' is from Figure 6.1 in: Design and Analysis of Experiments (Montgomery, 2013)
- `bacterial-growth.csv' is from Problem 6.8 in: Design and Analysis of Experiments (Montgomery, 2013)
- `circuit-board.csv' is from Problem 6.5 in: Design and Analysis of Experiments (Montgomery, 2013)

## Resources
Here are some resources that may be helpful:
- A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)
