# Machine Learning - Universidade de Aveiro - Python 

##  Setup

### Installing Python3 
Python 3.6 is the current latest release.

#### For Windows and Mac OS users
Check [this page](https://www.python.org/downloads/release/python-360/) (the download links are in the botton of the page). Download the appropriate (the executable installer for Windows is a good one) installer and run it (during the installation select the `Add Python to PATH` option).


#### For Linux users 
Usually the newer distros come with Python 2 and 3 pre-installed. Run `python3 -V` in your Terminal to check the current version. If Python is not present, run `sudo apt-get install python3` to install it.

After the installation open a console window (Command Prompt on Win and Terminal on Mac/Ubuntu) and run the `python` command. You should see something like that. 

<img src="http://i.imgur.com/ZaxZk6A.png" width="400" alt="Python on Windows">

Python is up and running!

### Installing Pip (Python Package Manager)
Pip helps us to install different Python libraries. 

#### For Windows users
The Python installer should add Pip as well by default. Test it with the `pip -V` command in Command Prompt. If Pip is not present, please check [this link](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation#pip-install).

#### For MacOS\Linux users
Check if Pip is installed by running `pip -V` in the Terminal. If it's not, run `sudo easy_install pip`.

### Installing NumPy 

[NumPy is the fundamental package for scientific computing with Python](http://www.numpy.org/) and we will use it for matrix calculations. You can install NumPy by running `pip install numpy` in the Terminal/Command Prompt. 

## Matrix operations with NumPy

Let's explore the functionality of NumPy. 
Check the examples below and run them one by one. You can use the Python prompt in the Terminal/Command Prompt, IDLE (the default Python shell) or your favourite text editor to write and execute Python code. 

```python
import numpy as np

# create a simple 2x2 matrix using NumPy
m = np.matrix('1 2; 3 4');

# create a vector (one-dimensional array)
v = np.array([1,3,2])

# get matrix's/vector's dimension
v.shape

# turn the vector to 2D matrix
v = v[None, :]
v.shape

# another way to create a matrix
matrixA = np.array([1,1,2,3,5,8,13,21,34]).reshape(3,3)

# creating a vector as an arithmetic series
vectorB = np.arange(0,20,2)

# reshape it to a matrix
matrixB = vectorB.reshape(2,5)

# you can perform +,-,*,/** operations using a scalar and a matrix
matrixB * 2
matrixB / 2
matrixB + 10
matrixB - 20
matrixB ** 2

# Matrix Operations

# let's create two 2D matrices with random values
matrixX = np.random.random_integers(0,10, (3,3))
matrixY = np.random.random_integers(0,10, (3,3)) 

# addition
matrixZ = matrixX + matrixY

# subtractions
matrixZ = matrixX - matrixY

# multiplication
matrixZ = matrixX * matrixY

# or
matrixZ = np.dot(matrixX, matrixY)

# transpose
matrixZ = np.matrix.transpose(matrixX)

# find the determinant of a matrix
det = np.linalg.det(matrixX)
```

## Solving Linear Regression problem using Python 

### Setuping matplotlib
[Matplotlib is a Python 2D plotting library]() we are going to use to display our data. In order to install it, we will use pip again, so just run `pip install matplotlib`.

### Exercises

**Objective:** Impelement linear regression and get to see how it works on data.
First, you need to download the starter code to the directory where you wish to complete the exercise. 

**Files included**
* `linearregression1var.py` - Python script that guides you through the exercise (the main program)
* `data.txt` - Training dataset

Let's say that you want to open a new store and you are considering different cities for the opening. You have already collected some data you want to use. In the first part of the exercise you are going to implement a linear regression model that is going to help you evaluate the parameters of a function that predicts profits for the new store. 

1. **Load and visualize the data**   
The file `data.txt` contains the data we have so far. The first column is the population of the city and the second column is the profit of having a store in that city. A negative value for profit indicates a loss. Load the information into the `data` variable and initialize `X` and `y` afterwards. We will create a scatter plot in order to visualize the data. It should look something like that:

<img src="http://i67.tinypic.com/11j99pj.png" width="400" alt="Scatter plot">

2. **Cost function and Gradient descent**  
We can fit our linear regression parameters to our dataset using gradient descent. The parameters of your model are the θ values. These are the values you will adjust to minimize cost J(θ). One way to do it is to use the batch gradient descent algorithm. In batch gradient, each iteration performs the update. With each step of gradient  descent, your parameters θ, come close to the optimal values that will achieve the lowest cost J. For our initial inputs we start with our initial fitting parameters θ, our data and add another dimmension to our data  to accommodate the θo intercept term. As also our learning rate alpha to 0.01.

3. **Predictions**  
Predict values for population sizes of 35 000 and 70 000 and plot the values using scatter plot. It should look something like that:
<img src="http://i67.tinypic.com/anefsj.png" width="400" alt="Scatter plot">
 
