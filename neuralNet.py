import matplotlib.pylab as plt
import numpy as np
import sys
import os
from LogisticRegression import get_data

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
            Type of the cost function used ['mse', 'log']
        regularization: string
            Type of the regularization used ['l1', 'l2']
        lamda: float
            regularization strength
        param: lamb: strength of regularization
        """
        self.xData = X
        self.yData = y
        self.N = X.shape[0]
        self.cost_func = cost_function
        self.regularization = regularization
        self.lamda = lamda

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

        xData = self.xData
        yData = self.yData

        if shuffle:
            xData = self.xData.iloc[ np.random.permutation(self.N) ]
            yData = self.yData.iloc[ np.random.permutation(self.N) ]

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

    def cost_function(self, y, ypred):
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
            cost =  0.5 / y.reshape(-1, 1).shape[0] * np.sum( (y.reshape(-1, 1) - ypred) ** 2)

        if self.cost_func == 'log':
            cost = -0.5 / y.reshape(-1, 1).shape[0] * np.sum( np.log(ypred[np.arange(ypred.shape[0]), y.reshape(-1, 1).flatten()]) )

        if self.regularization == 'l2':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(self.Weights[key]**2)

        elif self.regularization == 'l1':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(np.abs(self.Weights[key]))

        return cost


    def cost_function_derivative(self, y, ypred):
        """
        Takes the derivative of the selected cost function - same setup as with the activations
        INPUT: 
        ---------
        y: numpy ndarray
            correct labels
        ypred: numpy ndarray
            predicted targets

        OUTPUT:
        ---------
        returns the calculated derivative of the costfunction
        """

        if self.cost_func == 'mse':
            return -1.0 / y.shape[0] * (y - ypred.flatten())

        elif self.cost_func == 'log':
            ypred[ np.arange(ypred.shape[0]), y.flatten() ] -= 1
            return 1.0 / y.shape[0] * ypred

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

        self.A['A0'] = x
        for i in range(self.nLayers):
            z = np.dot( self.A['A'+str(i)], self.Weights['W'+str(i+1)] ) + self.Biases['B'+str(i+1)]
            a = self.activation_function(z, self.activations[i])
            self.A['A'+str(i+1)] = a

        if self.cost_func == 'log':
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

        for i in range(self.nLayers, 0, -1):
            if i == self.nLayers:
                c = self.cost_function_derivative(yTrue, self.output)
            else:
                try:
                    c = np.dot(c, self.Weights['W'+str(i+1)].T)
                except: # for the very first iteration
                    c = np.dot(c, self.Weights['W'+str(i+1)])
                c = c * self.activation_function(self.A['A'+str(i)], self.activations[i-1], derivative=True)

            grad_w = np.dot(self.A['A'+str(i-1)].T, c)
            grad_b = np.sum(c, axis= 0)

            self.Weights_grad['dW'+str(i)] = grad_w
            self.Biases_grad['dB'+str(i)] = grad_b

            if self.regularization == 'l2':
                self.Weights['W'+str(i)] -= self.eta * (grad_w + self.lamb*self.Weights['W'+str(i)])

            elif self.regularization == 'l1':
                self.Weights['W'+str(i)] -= self.eta * (grad_w + self.lamb*np.sign(self.Weights['W'+str(i)]))

            else:
                #updateSize = (self.eta * grad_w).reshape(-1, 1)
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
        tau: float
            learning rate
        n_print: int
            stepsize of when to calculate the errors and print them
        """

        if tau == 'schedule':
            t0 = 5 ; t1 = 50
            eta = lambda t : t0/(t + t1)
        else:
            eta = lambda t : tau

        num_batch = int(self.nTrain // batchSize)

        self.convergence_rate = {'Epoch': [], 'Test Accuracy': []}
        for epoch in range(epochs +1):
            indices = np.random.choice(self.nTrain, self.nTrain, replace=False)

            for b in range(num_batch):
                self.eta = eta(epoch*num_batch+b)
                # get the batches 
                batch = indices[b*batchSize : (b+1)*batchSize]
                yBatch = self.yTrain[batch]

                #self.xTrain.set_index(np.arange(self.xTrain.shape[0]), inplace=True)
                xBatch = self.xTrain.iloc[batch]

                # propagate them through the network twice (forward and backward)
                self.feed_forward(xBatch)
                self.backpropagation(yBatch)

            if epoch == 0 or epoch % n_print == 0:
                ypred_train = self.feed_forward(self.xTrain, isTraining=False)
                ypred_test  = self.feed_forward(self.xTest, isTraining=False)
                trainError  = self.cost_function(self.yTrain, ypred_train)
                testError   = self.cost_function(self.yTest, ypred_test)
                print("[ERROR]: {} epoch from {}: Training Error:  {}, Test Error  {}".format(epoch, epochs+1, trainError, testError))

                if self.cost_func == 'log':
                    trainAcc = self.accuracy(self.yTrain, ypred_train)
                    testAcc = self.accuracy(self.yTest, ypred_test)
                    print("[ACCURACY]: {} epoch from {}: Training Acc:  {}, Test Acc  {}".format(epoch, epochs+1, trainAcc, testAcc))
                    self.convergence_rate['Epoch'].append(epoch)
                    self.convergence_rate['Test Accuracy'].append(testAcc)

## TESTING MY BABY 
if __name__ == '__main__':
    # read the dataset
    cwd = os.getcwd()
    filename = "\\default of credit card clients.xls"
    filePath = cwd + filename
    X, y = get_data(filePath, standardized=False, normalized=False)
    neuralNet = NeuralNet(X, y, nodes=[23, 64, 64, 128, 1], activations=['relu', 'relu','relu', 'sigmoid'])
    neuralNet.split_data(test_size=0.2)
    neuralNet.trainingNetwork(epochs=250, batchSize=128, tau=0.01)

    ypred_test = neuralNet.feed_forward(neuralNet.xTest, isTraining=False)
    acc = neuralNet.accuracy(neuralNet.yTest, ypred_test)
    print("Accuracy: {}".format(acc))