# add the necessary dependencies
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

def compute_cost(X, y, theta):
	# get the number of training examples
	m = y.size
	
	# calculate the predictions and the errors
	predictions = 
	sqErrors = 
	
	# compute the cost
	J = 
	
	return J

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

		# compute the gradients
		errors_x1 = 
		errors_x2 = 

		theta[0, 0] = 
		theta[1, 0] = 

		J_history[i, 0] = 
	
	return theta, J_history

# == Task 1 ===================
	
# Load the dataset from the text file
data = 

# Initialize X and y
X = 
y = 

# Plotting the data
scatter(X, y, marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
# show()

# ===================

# == Task 2 ===================

# get the number of training examples
m = 

# add a column of ones to X (interception data)
it = 

# initialize theta
theta = 

# initialize the number of iterations to 1500 and alpha to 0.01
iterations =
alpha =  

# show the initial cost
print(compute_cost(it, y, theta))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print(theta)

# =================== 

# == Task 3 ===================

#Predict values for population sizes of 35,000 and 70,000
predict1 = 
print('For population = 35,000, we predict a profit of %f' % (predict1 * 10000))
predict2 = 
print('For population = 70,000, we predict a profit of %f' % (predict2 * 10000))

#Plot the results
result = it.dot(theta).flatten()
plot(data[:, 0], result)
show()

# =================== 