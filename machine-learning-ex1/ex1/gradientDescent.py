def batch_gradient_descent(loss_func, data, params, num_epochs):
	for i in range(num_epochs):
		gradient = evaluate_gradient(loss_func, data, params) #this is a vector
		params = params - alpha * gradient #vector subtraction

def stochastic_gradient_descent(loss_func, data, params, num_epochs) :
	for i in range(num_epochs):
		shuffle_data(data)
		for sample in data:
			gradient = evaluate_gradient(loss_func, sample, params)
			params = params - alpha * gradient

def minibatch_gradient_descent(loss_func, data, params):
	for i in range(num_epochs):
		shuffle_data(data)
		for batch in getBatches(data):
			gradient = evaluate_gradient(loss_func, batch, params)
			params = params - alpha * gradient

def momentum_update(gamma=0.9, params, eta, data):
	velocity_t = (gamma * velocity_t_minus_1) + eta * evaluate_gradient(loss_func, data, params)
	return params - velocity_t

def nag_update():
	probable_params = params - gamma * velocity_t_minus_1
	velocity_t = (gamma * velocity_t_minus_1) + eta * evaluate_gradient(loss_func, data, probable_params)
	return params - velocity_t

def adagrad(eta=0.01):
	#uses different learning rate for each of the params
	def createMatrix(params, loss_func) :
		G = emptyMatrix(size(params))
		G(i,i) = doSumOfSquaresOfGradient(i, time=t)
		return G
	matrix_G = createMatrix(params, loss_func)
	learning_rate_matrix = eta / sqrt(G + epsilon)
	return params - elementWiseProduct(learning_rate_matrix, evaluate_gradient(loss_func, data, params_t))

def adadelta(gamma=0.9, eta=0.01):
	# instead of storing all the part gradients lets keep a window of size w
	# we can compute it on the go in online fashion
	E_of_grad[t] = gamma*E_of_grad[t-1] + (1-gamma) * square(evaluate_gradient(loss_func, data, params_t))
	E_matrix = createMatrix(E_of_grad[t])
	RMS[t] = square(E_matrix + epsilon)
	return params - elementWiseProduct(eta/RMS, evaluate_gradient(loss_func, data, params_t))

def adadeltaV2(gamma=0.9):
	return adadelta(gamma, eta=RMS[t-1])

def RMSprop(eta):
	return adadelta(gamma=0.9, eta);



