import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import sys
import os
from LogisticRegression import get_data
from sklearn.neural_network import MLPClassifier # benchmarking

class NeuralNet():

    def __init__(self, xData, yData, nodes=[23, 10, 1], activations=['sigmoid', None], cost_function='mse', regularization=None, lamda=0.0):
        """
        This method is used to create a Multi Layered Perceptron (MLP)
        
        INPUT:
        ---------
        X: pandas.DataFrame
            Data for training and testing respectively
        y: pandas.DataFrame
            Corresponding target values to the data above
        nodes: list of integers
            Number of neurons/ units in each layer, the length of the lists represents the depth of the network (length 3 = 3  layers)
            First number needs to be the size of the input and last element is the output we want
            Last element 1 for regression or binary classification, otherwise multi-class classification
        activations: list of strings
            activation functions for each of the layers respectively
            However, no activation for the first layer - so has the shape: len(nodes) -1  
        cost_function: string
            Type of the cost function used ['mse', 'cross_entropy']
        regularization: string
            Type of the regularization used ['l1', 'l2']
        lamda: float
            regularization strength
        param: lamb: strength of regularization
        """
        self.xData = xData
        self.yData = yData
        self.N = xData.shape[0]
        self.cost_func = cost_function
        self.regularization = regularization
        self.lamda = lamda
        self.learning_rate = []

        if (len(activations) != len(nodes) -1) :
            raise ValueError("You provided a wrong input! \nUsage should be: nodes = [input, hidden layer1, ..., hidden layer n, output].\nActiavations need to be 1 smaller (len(nodes) - 1)!")

        self.nodes = nodes
        self.activations = activations
        self.nLayers = len(activations)

        self.split_data(folds=10, test_size=0.2, shuffle=True)
        self.initialize_weights_biases()

    def split_data(self, folds=None, test_size=None, shuffle=False):
        """
        Splits the data into training and test given the test_size
        INPUT:
        ---------
        folds: int
            number of folds that you want to split the entire data into
        test_size: float
            values from [0.0, ..., 1.0] representing the size of the test dataset.
        shuffle: boolean
            Flag stating whether to shuffle or not the entire dataset.
        """

        if folds == None and test_size == None:
            raise ValueError("You need to give either a test_size fraction or the number of folds")

        if type(self.xData).__module__ == np.__name__:
            self.xData = pd.DataFrame(self.xData)
            self.yData = pd.DataFrame(self.yData)
        else:
            xData = self.xData
            yData = self.yData

        if shuffle:
            try:
                xData = self.xData.iloc[ np.random.permutation(self.N) ]
                yData = self.yData.iloc[ np.random.permutation(self.N) ]
            except: 
                xData = self.xData[ np.random.permutation(self.N) ]
                yData = self.yData[ np.random.permutation(self.N) ]

        if folds != None:
            xFolds = np.array_split(xData, folds, axis = 0)
            yFolds = np.array_split(yData, folds, axis = 0)

            self.xFolds = xFolds
            self.yFolds = yFolds

        if test_size != None:
            nTest = int( np.floor(test_size*self.N) )
            xTrain = xData[ : -nTest]
            xTest = xData[ -nTest : ]

            yTrain = yData[ : -nTest ]
            yTest = yData[ -nTest : ]

            self.xTrain = xTrain
            self.xTest  = xTest
            self.yTrain = yTrain
            self.yTest  = yTest
            self.nTrain = xTrain.shape[0]
            self.nTest  = xTest.shape[0]

    def initialize_weights_biases(self):
        """
        Initializes weights and biases for all layers
        INPUT: 
        ---------
        None
        """

        self.Weights        = {}
        self.Biases         = {}
        self.Weights_grad   = {}
        self.Biases_grad    = {}
        self.A              = {}

        for i in range(len(self.activations)):

            if self.activations[i] == 'sigmoid':
                r = np.sqrt(6.0 / (self.nodes[i] + self.nodes[i+1]))
                self.Weights['W'+str(i+1)] = np.random.uniform(-r, r, size=(self.nodes[i], self.nodes[i+1]))
                self.Biases['B'+str(i+1)]  = np.random.uniform(-r, r, self.nodes[i+1])
            elif self.activations[i] == 'tanh':
                r = 4.0 * np.sqrt(6.0 / (self.nodes[i] + self.nodes[i+1]))
                self.Weights['W'+str(i+1)] = np.random.uniform(-r, r, size=(self.nodes[i], self.nodes[i+1]))
                self.Biases['B'+str(i+1)]  = np.random.uniform(-r, r, self.nodes[i+1])
            elif self.activations[i] == 'relu':
                self.Weights['W'+str(i+1)] = np.random.normal(size=(self.nodes[i], self.nodes[i+1])) * np.sqrt(2.0 / self.nodes[i])
                self.Biases['B'+str(i+1)]  = np.random.normal(self.nodes[i+1]) * np.sqrt(2.0 / self.nodes[i])
            else :
                self.Weights['W'+str(i+1)] = np.random.normal(size=(self.nodes[i], self.nodes[i+1]))
                self.Biases['B'+str(i+1)]  = np.random.normal(self.nodes[i+1])

            self.Weights_grad['dW'+str(i+1)] = np.zeros_like(self.Weights['W'+str(i+1)])
            self.Biases_grad['dB'+str(i+1)] = np.zeros_like(self.Biases['B'+str(i+1)])


    def activation_function(self, x, act_func, derivative=False):
        """
        Calculate the activation function
        INPUT:
        ---------
        x: numpy ndarray
            data that should be chased through the activation function
        act_func: string
            string stating the activation function wanted ['sigmoid', 'relu', 'tanh']
        derivative: boolean
            flag indicating whether the derivative should be retrieved or not

        OUTPUT:
        ----------
            returns the desired activation function results.
        """
        if act_func == 'sigmoid':
            with np.errstate(divide='ignore'): # some runtimeWarnings
                return x * (1 - x) if derivative else 1/(1+np.exp(-x))

        elif act_func == 'tanh':
            return 1 - x**2 if derivative else np.tanh(x)

        elif act_func == 'relu':
            return 1*(x >= 0) if derivative else x * (x >= 0)

        elif act_func == None: # identity
            return 1 if derivative else x

        else:
            raise ValueError("You have to select a proper activation function of ['sigmoid', 'relu', 'tanh', None].\n Given: {}".format(act_func))


    def softmax(self, act):
        """
        calculates the softmax function and returns it
        """

        # Subtraction of max value for numerical stability
        act_exp = np.exp(act - np.max(act))
        self.act_exp = act_exp
        return act_exp/np.sum(act_exp, axis=1, keepdims=True)

    def cost_function(self, y, ypred, derivative=False):
        """
        calculate the cost of the initialized value
        INPUT:
        --------
        y: numpy ndarray
            correct targets given by the dataset
        ypred: numpy ndarray
            predicted targets given the training or testing features

        OUTPUT:
        ----------
        cost: float
            cost of the function specified
        """

        if self.cost_func == 'mse':
            if derivative: 
                try: 
                    deriv = ypred.flatten() - y
                except:
                    deriv = ypred - y
                return deriv
            else:
                y = np.array(y)
                norm = ( (y.flatten() - ypred)**2 ).mean() * 0.5
                return norm #deriv if derivative else norm

        if self.cost_func == 'cross_entropy':
            if derivative: 
                try: 
                    deriv = ypred.flatten() - y
                except: 
                    deriv = ypred - y
                return deriv
            else: 
                norm = -np.sum(y.flatten() * np.log(ypred) + (1 - y.reshape(-1, 1))*np.log(1-ypred)) / ypred.shape[0]
                return norm # deriv if derivative else norm
            #https://deepnotes.io/softmax-crossentropy#derivative-of-cross-entropy-loss-with-softmax
            #-0.5 / y.reshape(-1, 1).shape[0] * np.sum( np.log(ypred[np.arange(ypred.shape[0]), y.reshape(-1, 1).flatten()]) )

        if self.regularization == 'l2':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(self.Weights[key]**2)
            return cost

        if self.regularization == 'l1':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(np.abs(self.Weights[key]))
            return cost


    def accuracy(self, y, ypred):
        """
        calculates the accuracy = number of correctly classified classes
        INPUT:
        --------
        y: numpy ndarray
            correct labels
        ypred: numpy ndarray
            predicted targets

        OUTPUT:
        ---------
        returns the calculated accuracy
        """
        cls_pred = np.argmax(ypred, axis=1)
        return 100.0 / y.shape[0] * np.sum(cls_pred == y)


    def feed_forward(self, x, isTraining = True):
        """
        Calculating the feed forward pass of the neural network
        INPUT:
        --------
        x: numpy.ndarray
            Data given to train our network
        isTraining: boolean
            Flag to state whether we are in training or not
        
        OUTPUT:
        --------
        a: numpy.ndarray
            output after forwardpass (activation(W*x + b) )
        """
        # input
        self.A['A0'] = x
        for i in range(self.nLayers):
            z = np.dot( self.A['A'+str(i)], self.Weights['W'+str(i+1)] ) + self.Biases['B'+str(i+1)]
            a = self.activation_function(z, self.activations[i])
            self.A['A'+str(i+1)] = a

        if self.cost_func == 'cross_entropy':
            a = self.softmax(a)

        if isTraining:
            self.output = a
        else:
            return a


    def backpropagation(self, yTrue=None):
        """
        doing the funny backpropagation
        INPUT:
        --------
        yTrue: numpy ndarray
            Correct values of the targets y
        """
        if yTrue is None:
            yTrue = self.yTrain

        # work the way from back to front, start at the output now
        for i in range(self.nLayers, 0, -1): # reverse range
            if i == self.nLayers:
                deltaCost = self.cost_function(yTrue, self.output, derivative=True)
                deltaCost = deltaCost.values.reshape(-1, self.nodes[len(self.nodes) -1])
                # delta.shape = (BatchSize, lastNode)
            else:
                #print(c.shape, self.Weights['W'+str(i+1)].shape)
                try:
                    deltaCost = np.dot(deltaCost, self.Weights['W'+str(i+1)].T)
                except: 
                    deltaCost = np.dot(deltaCost, self.Weights['W'+str(i+1)])
                deltaCost = deltaCost * self.activation_function(self.A['A'+str(i)], self.activations[i-1], derivative=True)

            grad_w = np.dot(self.A['A'+str(i-1)].T, deltaCost)
            grad_b = np.sum(deltaCost, axis= 0)

            self.Weights_grad['dW'+str(i)] = grad_w
            self.Biases_grad['dB'+str(i)] = grad_b

            if self.regularization == 'l2':
                self.Weights['W'+str(i)] -= self.eta * (grad_w + self.lamb*self.Weights['W'+str(i)])

            elif self.regularization == 'l1':
                self.Weights['W'+str(i)] -= self.eta * (grad_w + self.lamb*np.sign(self.Weights['W'+str(i)]))

            else: # update step
                try:
                    self.Weights['W'+str(i)] -= (self.eta * grad_w).reshape(-1, 1)
                except:
                    self.Weights['W'+str(i)] -= (self.eta * grad_w)

            self.Biases['B'+str(i)] -= self.eta * grad_b


    def trainingNetwork(self, epochs=1000, batchSize=200, tau=0.01, n_print=100):
        """
        Training the neural network by utilizing forward and backward propagation
        INPUT:
        --------
        epochs: int
            number of epochs to do
        batchSize: int
            size of the batches -> one epoch = len(Data) / batchSize
        tau: float or string
            learning rate type and potentially the normal learning rate
        n_print: int
            stepsize of when to calculate the errors and print them
        """

        if tau == 'schedule':
            t0 = 5 ; t1 = 50
            eta = lambda t : t0/(t + t1)
        else:
            eta = lambda t : tau

        num_batch_per_epoch = int(self.nTrain // batchSize)

        self.convergence_rate = {'Epoch': [], 'Test Accuracy': []}
        for epoch in range(epochs +1):
            indices = np.random.choice(self.nTrain, self.nTrain, replace=False)

            for batch in range(num_batch_per_epoch):
                self.eta = eta(epoch * num_batch_per_epoch + batch)
                # get the batches 
                batch = indices[batch * batchSize : (batch+1)*batchSize]
                # somehow running into index errors always ...
                self.yTrain.set_index(np.arange(self.yTrain.shape[0]), inplace=True)
                self.xTrain.set_index(np.arange(self.xTrain.shape[0]), inplace=True)

                #print(self.yTrain.index)

                yBatch = self.yTrain.iloc[batch]
                xBatch = self.xTrain.iloc[batch]

                # propagate them through the network twice (forward and backward)
                self.feed_forward(xBatch)
                self.backpropagation(yBatch)

                #self.learning_rate.append(self.eta)

            if epoch == 0 or epoch % n_print == 0:
                ypred_train = self.feed_forward(self.xTrain, isTraining=False)
                ypred_test  = self.feed_forward(self.xTest, isTraining=False)
                trainError  = self.cost_function(self.yTrain, ypred_train)
                testError   = self.cost_function(self.yTest, ypred_test)
                #print("[ERROR]: {} epoch from {}: Training Error:  {}, Test Error  {}".format(epoch, epochs+1, trainError, testError))

                if self.cost_func == 'cross_entropy':
                    trainAcc = self.accuracy(self.yTrain, ypred_train)
                    testAcc = self.accuracy(self.yTest, ypred_test)
                    #print("[ACCURACY]: {} epoch from {}: Training Acc:  {}, Test Acc  {}".format(epoch, epochs, trainAcc, testAcc))
                    self.convergence_rate['Epoch'].append(epoch)
                    self.convergence_rate['Test Accuracy'].append(testAcc)

## TESTING MY BABY 
if __name__ == '__main__':
    # read the dataset
    cwd = os.path.join(os.getcwd(), "data")
    filename = "default of credit card clients.xls"
    filePath = os.path.join(cwd, filename)
    X, y = get_data(filePath, standardized=False, normalized=False)
    # note that the activation functions need to be 1 smaller than the node length
    activation_functions    = ['relu', 'relu', 'relu', 'relu', None]
    neural_setup            = [23, 128, 64, 32, 32, 1] # first needs to be 23 and last needs to be 1
    neuralNet = NeuralNet(X, y, nodes=neural_setup, activations=activation_functions, cost_function='cross_entropy')
    neuralNet.split_data(test_size=0.2)
    neuralNet.trainingNetwork(epochs=100, batchSize=64, tau=0.01)

    ypred_test = neuralNet.feed_forward(neuralNet.xTest, isTraining=False)
    acc = neuralNet.accuracy(neuralNet.yTest, ypred_test)
    print("My      Accuracy: {}".format(acc))

    clf = MLPClassifier(hidden_layer_sizes=neural_setup, learning_rate_init=0.01).fit(neuralNet.xTrain, neuralNet.yTrain)
    print("SKLearn Accuracy: {}".format(clf.score(neuralNet.xTest, neuralNet.yTest)))

## ------------------------------ OTHER CODE TO CHECK
'''
    def feed_forward(self):
        self.z1 = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a1 = sigmoid(self.z1)

        self.z2 = np.matmul(self.a1, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z2)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        z1 = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a1 = sigmoid(z1)

        z2 = np.matmul(a1, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z2)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a1 * (1 - self.a1)

        self.output_weights_gradient = np.matmul(self.a1.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
'''