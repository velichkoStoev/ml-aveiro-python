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

