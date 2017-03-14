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
Check the file `matrices.py` and run the examples there one by one. You can use the Python prompt in the Terminal/Command Prompt, IDLE (the default Python shell) or your favourite text editor to write and execute Python code. 
