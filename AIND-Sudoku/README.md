# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Here are the steps we take to find the naked twins.
    1. Iterate through each unit in the unit-list 
    2. For each unit create a dictionary where keys are the boxes with two choices and the values are the number of times the choices occurs in the unit
    3. From the dictionary select the box values which occur more than once, these are the naked twins in the unit.
    4. Store the twins in a dictionary keyed on the unit id.
    4. Iterate through the twins dictionary and the units removing the digits, forming the respective twins, from the boxes in the unit


# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: To solve the Diagonal sudoku we just add two more (the boxes found along the two diagonals) units to the unit-list. By adding the diagonal units to the unit-list the
   diagonal constraint is evaluated like the other existing constraints, without any additional code changes. The boxes for the diagonal units are found easily using the
   functions gen_diag() and get_diagonal_units() functions

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.
