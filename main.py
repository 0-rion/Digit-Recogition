import numpy as np

class Network(object):

    # initialising function
    def __init__(self, sizes): # sizes defines the number of neurons in each layer, sizes = [2, 3, 1]
        self.num_layers = len(sizes)
        self.sizes = sizes
        # initialise biases for each layer (input layer omitted)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
        # initialise weights for each neuron (input layer omitted)
        # y - the number of neuron in current layer
        # x - the number of neuron in previous layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def layer_output(self,layer_num,a_in):
        # layer_num starts at 0 indicating the output of second layer (first hidden layer), ends at num_layers - 1 (output layer)
        w = self.weights[layer_num]
        b = self.biases[layer_num]
        a_out = sigmoid(np.matmul(w,a_in) + b)
        return a_out

    def output(self,a_in):
        for i in range(self.num_layers - 1):
            a_in = self.layer_output(self,i,a_in)
        return a_in

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data = None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  
        
        "training_data" is a list of tuples "(x, y)" representing the training inputs and the desired outputs.  
        "epochs" is the number epochs
        "mini_batch_size" is the size of one batch
        "eta" is the learning rate

        If "test_data" is provided then the network will
        be evaluated against the test data after each epoch, and 
        partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        # check is test_data is provided
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            # partition data into mini batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # stochastic gradient descent is used here
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""

        # It's more efficient to compute the gradients for all 
        # training examples in a mini-batch simultaneously using matrix

        # initialise the gradient of each weight and bias for this mini batch
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # backpropagation algorithm
            # parallel iteration with zip()
            # summing weight and bias derivative for every data in mini batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # update weigths and biases by finding the average weight and bias derivative
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                        for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # is np.dot a matrix multiplication or dot product? it should be matrix multiplication
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # find error of last layer
        # "\" for statement contiune to next line
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # weight and bias derivative of last layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # change to matrix multiplication
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            # find error for each layer
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



