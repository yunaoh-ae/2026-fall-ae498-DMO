# HW5 - Optimization and Genetic Algorithms

## Problem 1: KKT Conditions Concept Question

Use the KKT necessary and sufficient conditions to check if the point $x=[1,5]^T$ is an optimum to the following problem (show your work):

Minimize:	$f = x_1^2-x_2$

Subject to:

$g_1=x_1-1≥0$

$g_2=26-x_1^2-x_2^2≥0$

$h_1=x_1+x_2-6=0$

You can submit your solution in the `hw4.ipynb` file or a separate scanned `hw4.pdf` file.

## Problem 2: Canonical GA Concept Question

Assume we are working with a canonical, binary GA and we want to minize the following function:

$J(x_1,x_2 )=(x_1-10)^2+[5(x_2+5)]^2-200$.

1. What are two possible fitness functions?

1. Assume the fitness values of a population are: 14, 85, 25, 79, 93, 21, 29, 76.

    1. Using proportionate selection, what is the expected number of times each design will be in the selected mating pool?

    1. Using ranked selection, what is the expected number of times each design will be in the selected mating pool?

You can submit your solution in the `hw4.ipynb` file or a separate scanned `hw4.pdf` file.

## Problem 3: Optimize a Function Using a GA

Your goal is to use a GA to optimize the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function), with $a = 1$ and $b = 100$. Suggested variable limits are $-2 \leq x_1 \leq 2$ and $-1 \leq x_2 \leq 3$, with 32 settings (levels) for each variable. The true minimum of this function is $[1, 1]$.

Optimize with the fitness function being the negative of the Rosenbrock function, such that,

$J = -[(1-x_1 )^2+100(x_2-x_1^2 )^2 ]$.

To this, please update the `hw4.ipynb` Jupyter notebook file to:
- optimize the Rosenbrock function using a GA 100 times (i.e., using 100 trials)
- show and discuss the best solution you achieve, relative to the true minimum, over iterations for at least one trial
- show and discuss a histogram of the errors in your optimal solutions (across trials)
    - you can measure error as the Euclidean distance between each of your minimum solutions and the true minimum

I should be able to run your notebook to recreate your results. You can (and probably should) use existing libraries for this task.

## Submission

1. Submit your updated `hw4.ipynb` files to your GitHub repository.
    1. You can do this through GitHub in a web browser by: click on your repository > click "Add file" > click Upload files > drag your updated files > click "Commit changes".
1. Submit your assignment on Gradescope by submitting your GitHub repository.

## Resources
Here are some resources that may be helpful:
- A classic reference on writing code: [The Art of Readable Code (Boswell and Foucher, O'Reilly, 2012)](https://mcusoft.files.wordpress.com/2015/04/the-art-of-readable-code.pdf)
