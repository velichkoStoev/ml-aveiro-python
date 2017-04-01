from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1.
    '''
    mean_r = []
    std_r = []

    X_norm = X

    n_c = X.shape[1]
    for i in range(n_c):
       # Your code goes here ...

    return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
    '''
    Compute cost for linear regression - you can use the function you wrote in the previous example
    '''


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
	
	# get the number of training examples
    m = 
	
	# initialize J_history to zeros
    J_history = 

    for i in range(num_iters):

        predictions = 

        theta_size = 

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = 

            theta[it][0] = 

        J_history[i, 0] = 

    return theta, J_history

# == Task 1 ===================	
	
# Load the dataset from the text file
data = 

# Initialize X and y

X = 
y = 


#Plot the data
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25)]:
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('Size of the House')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price of the House')

plt.show()
'''

# ===================

# == Task 2 ===================

#number of training samples
m = 

y.shape = (m, 1)

#Scale features and set them to zero mean
x, mean_r, std_r = feature_normalize(X)

# ===================

# == Task 2 ===================

#Add a column of ones to X (interception data)
it = 

#Set the number of iterations to 100 and alpha to 0.01
iterations = 
alpha = 

#Initialize theta and run Gradient Descent
theta = 

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print (theta, J_history)

# Plot J_History
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

# ===================

# == Task 3 ===================

#Predict the price of a 1650 sq-ft 3 br house
price = 
print ('Predicted price of a 1650 sq-ft, 3 br house: %f' % (price))

# ===================
