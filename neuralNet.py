from sklearn.metrics import r2_score
from Activation import Activation
from math import ceil
import pandas as pd
import numpy as np
import warnings
import sklearn
import pickle
import copy
import time
import os


class NeuralNetwork :
    '''
    This Class implements a neural network (MLP) from scratch with all its necessary functions.
    '''
    def __init__(self, inputs=None,outputs=None, layers=None, neurons=None, activations=None, cost_function_name='mse', lmbda=0.0):
        self.inputs         = inputs
        self.outputs        = outputs
        self.layers         = layers
        self.neurons        = neurons
        self.activations    = activations
        self.lmbda          = lmbda

        self.weights        = None
        self.biases         = None

        self.cost_fct_name  = cost_function_name

        self.first_feedforward = True
        self.first_backprop    = True
        self.adam_initialized  = False

    def set(self, inputs=None, outputs=None, layers=None, neurons=None, activations=None):
        '''
        this function is basically a constructor and sets the values needed for the future
        '''
        self.inputs         = inputs        if (inputs      is not None) else self.inputs
        self.outputs        = outputs       if (outputs     is not None) else self.outputs
        self.layers         = layers        if (layers      is not None) else self.layers
        self.neurons        = neurons       if (neurons     is not None) else self.neurons
        self.activations    = activations   if (activations is not None) else self.activations


    def initializeWeight(self, n_in, n_out, activations):
        '''
        Initializes the Weights according to the Xavier Initialization or He Initialization (see papers below)
        
        INPUT:
        -------
        n_in: int
            Number of Inputs, which is also corresponding to the number of features/ predictor variables
        n_out: int
            Number of Outputs, 1 for Regression, C for number of classes you want to classify
        activations: string
            activation function name which is being used.

        OUTPUT:
        --------
            returns the values for the initialized weights
        '''
        # Xavier initializations (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
        if activations == 'sigmoid' :
            r = np.sqrt(6.0 / (n_in + n_out))
            return np.random.uniform(-r, r, size=(n_in, n_out))
        elif activations == 'tanh' :
            r = 4.0 * np.sqrt(6.0 / (n_in + n_out))
            return np.random.uniform(-r, r, size=(n_in, n_out))

        # He initializations (https://arxiv.org/pdf/1502.01852.pdf).
        elif activations == 'relu' or activations == 'leaky_relu' or activations == 'elu':
            return np.random.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in)

        else :
            return np.random.normal(size=(n_in, n_out))


    def addLayer(self, inputs=None, neurons=None, activations=None, alpha=None, outputs=None, output=False):
        """
        Function to setup the neural network architecture

        INPUT:
        --------
        inputs: int
            inputs/ features/ predictor variables into the network, default is None
        neurons: int
            Amount of Neurons (or Units) to be utilized in this respective layer, default is None
        activations: string
            states the activation function to be used of ['elu', 'relu', 'leaky_relu', 'sigmoid', 'tanh', 'softmax'], default is None
        alpha: float
            alpha value for the leaky_relu activation function, default is 0.01 (in Activation.py)
        outputs: int
            output layer size, i.e. number of classes to classify or 1 for Regression.
        output: boolean
            flag for the right last shape of the Biases and Weights, default is False

        OUTPUT: 
        ---------
            stores the results in the member variables
        """

        ## Check for the proper Input and raise errors if not satisfied
        if neurons is None :
            if self.neurons is None :
                raise ValueError("Please state the number of neurons in a layer")
            else:
                neurons = self.neurons
        if activations is None :
            if self.activations is None :
                warnings.warn("Please specify the activation functions for the hidden layers, using sigmoid now")
                self.activations = 'sigmoid'
                activations = self.activations
            else :
                activations = self.activations

        if self.weights is None :
            if inputs is None :
                if self.inputs is None :
                    raise ValueError("Please specify the number of inputs = number of features")
                inputs = self.inputs
            else:
                self.inputs = inputs 
            
            print( "Adding input layer with " + str(neurons) + " neurons, using " + str(activations))
            W = self.initializeWeight(inputs, neurons, activations)
            b = np.zeros(shape=(neurons, 1))
            f = Activation(function=activations, alpha=alpha)

            self.weights = [W]
            self.biases  = [b]
            self.act     = [f]
            
        elif output == True :
            if outputs is None :
                if self.outputs is None :
                    raise ValueError("Please specify the number of outputs = number of classes to be predicted, or 1 for Regression." )
                else :
                    outputs = self.outputs
            else :
                if self.outputs != outputs :
                    warnings.warn( "The number of outputs was earlier set to " + str(self.outputs) + ", but the value specified in method addLayer of " + str(outputs) + " replaces this.")
                    self.outputs = outputs

            print(  "Adding output layer with " + str(outputs) + " outputs and "  + str(activations))
            
            previousLayerNeurons = self.weights[-1].shape[1]
            W = self.initializeWeight(previousLayerNeurons, outputs, activations)
            b = np.zeros(shape=(outputs, 1))
            f = Activation(function = activations, alpha = alpha)
            
            self.weights.append(W)
            self.biases .append(b)
            self.act    .append(f)
        else :
            print("Adding layer with " + str(neurons) + " neurons using " + str(activations))
            previousLayerNeurons = self.weights[-1].shape[1]
            W = self.initializeWeight(previousLayerNeurons, neurons, activations)
            b = np.zeros(shape=(neurons,1))
            f = Activation(function = activations, alpha = alpha)

            self.weights.append(W)
            self.biases .append(b)
            self.act    .append(f)


    def layer(self, x, i) :
        """
        Training the neural network by utilizing forward and backward propagation

        INPUT:
        --------
        x: numpy.ndarray
            Entire data on which the model should be fitted, will be splitted in Train and Validation set.
        i: int
            corresponding layer number

        OUTPUT: 
        ---------
            returns the result from the next (hidden) layer
        """
        W = self.weights[i]
        b = self.biases[i]
        f = self.act[i]
        self.a[i+1] = f(np.dot(W.T, x) + b)
        return self.a[i+1]


    def __call__(self, x) :
        return self.network(x)

    def predict(self, x):
        return self.network(x)

    def forward_pass(self, x) :
        return self.network(x)

    def network(self, x):
        """
        This function performs the feed forward pass

        INPUT:
        --------
        x: numpy.ndarray
            data to be propagated through the network

        OUTPUT: 
        ---------
        x: numpy.ndarray
            Output value calculated through the forward propagation of the network
        """
        ## First Feedforward to set up the proper length of the hidden layer outputs
        if self.first_feedforward :
            self.a = [None]*(len(self.weights)+1)
            self.first_feedforward = False

        self.n_features, self.n_samples = x.shape
        ## First layer
        self.a[0] = x
        self.a[1] = self.act[0](np.dot(self.weights[0].T, x) + self.biases[0])
        x = self.a[1]

        ## All the other ones
        for i in range(1, len(self.weights)) :
            x = self.layer(x, i)
        return x

    def cost_function(self, ypred, y, cost_function='cross_entropy', derivative=False):
        '''
        Training the neural network by utilizing forward and backward propagation

        INPUT:
        --------
        ypred: numpy.ndarray
            predicted Outcome from the feed forward pass
        y: numpy.ndarray
            actual label and ground truth from data
        cost_function: string
            stating the name of the cost function to be used of ['mse', 'cross_entropy'], default is 'cross_entropy'
        derivative: boolean
            flag of whether we want the derivative of the respective cost function or not, default is False
        OUTPUT: 
        ---------
            returns the calculated cost function or derivative respectively
        '''
        if cost_function == 'cross_entropy':
            return (ypred - y) if derivative else -np.sum( y * np.log(ypred) + (1-y) * np.log(1-ypred) ) / ypred.shape[0]

        elif cost_function == 'mse':
            return (ypred - y) if derivative else ( (ypred - y)**2 ).mean() * 0.5

        else:
            raise ValueError("Please choose among ['cross_entropy', 'mse'] as a cost function.")


    def backpropagation(self, y, target) :
        """
        Performs the Backproagation through the network

        INPUT:
        --------
        y: numpy.ndarray
            predicted outcome of our network in the forward pass
        target: numpy.ndarray
            actual target from the data

        OUTPUT: 
        ---------
            stores the gradients in the member variables
        """

        ## First Backprop
        if self.first_backprop :
            self.delta      = [None]*len(self.weights)
            self.d_weights  = copy.deepcopy(self.weights)
            self.d_biases   = copy.deepcopy(self.biases)
            self.first_backprop = False

        self.delta[-1]      = ( self.cost_function(y, target, cost_function=self.cost_fct_name, derivative=True) * self.act[-1].derivative(self.a[-1].T) )
        self.d_weights[-1]  = np.dot(self.a[-2], self.delta[-1]) / self.n_samples
        self.d_biases[-1]   = np.mean(self.delta[-1], axis = 0, keepdims = True).T
        
        ## Backpropagate through the other layers
        for i in range(2, len(self.weights)+1) :
            self.delta[-i]      = ( np.dot(self.delta[-i+1], self.weights[-i+1].T) * self.act[-i].derivative(self.a[-i].T) )
            self.d_weights[-i]  = np.dot(self.a[-i-1], self.delta[-i]) / self.n_samples
            self.d_biases[-i]   = np.mean(self.delta[-i], axis = 0, keepdims = True).T

        ## Update the lovely, magical gradients of the respective weights
        for i, dw in enumerate(self.d_weights) :
            self.d_weights[i] = dw + self.lmbda * self.weights[i]

    def train_network(self, x, target, learning_rate=0.001, epochs=200, batch_size=200, val_size=0.1, val_stepwidth=10, optimizer='sgd', lmbda=None):
        """
        Training the neural network by utilizing forward and backward propagation

        INPUT:
        --------
        x: numpy.ndarray
            Entire data on which the model should be train_networkted, will be splitted in Train and Validation set.
        target: numpy.ndarray
            corresponding targets (y-values) for each of the inputs
        learning_rate: float
            learning rate size, default is 0.001
        epochs: int
            number of epochs to train ( 1 epoch = 1 entire run of all batches), default are 200
        batch_size: int
            size of the batches -> one epoch = len(Data) / batchSize
        val_size: float
            validation size of the training data, default is 0.1 which corresponds to 10% of the entire data
        val_stepwidth: int
            stepwidth of when the accuracy shall be calculated on validation set, default is after every 10th step
        optimizer: string
            describes the optimizer to be used of ['adam', 'sgd'], default is 'sgd'
        lmbda: float
            regularization parameter for the L2 regularization in the loss term, default is None

        OUTPUT: 
        ---------
            stores the results in the member variables
        """
        self.lmbda = lmbda if lmbda is not None else self.lmbda

        self.learning_rate = learning_rate
        self.best_loss  = None
        self.bestEpoch  = None
        self.best_param = None

        ## Set up the optimizer
        if optimizer == 'adam' :
            self.initializeAdam()
            self.optimizer = self.adam
        elif optimizer == 'sgd' :
            self.optimizer = self.sgd
        else :
            raise ValueError("The optimizer " + str(optimizer) + " is not supported. Please use either one of ['adam', 'sgd']")

        self.n_features, self.n_samples = x.shape
        x, target = sklearn.utils.shuffle(x.T, target.T, n_samples = self.n_samples)
        x = x.T
        target = target.T

        if not (self.n_features == self.inputs) :
            raise ValueError("Features in input data doesn't match the network!")

        ## Train test split of SKLearn can also be used here to reduce some more lines of code
        self.n_val              = int(round(val_size*self.n_samples))
        try:
            self.X_val              = x.values[:, : self.n_val]
            self.X_train            = x.values[:, self.n_val + 1 : ]
        except: 
            self.X_val              = x[:, : self.n_val]
            self.X_train            = x[:, self.n_val + 1 : ]

        if target.ndim == 1:
            target = target.reshape((1, -1))
        self.target_validation  = target[:, : self.n_val ]
        self.target_train       = target[:, self.n_val + 1 : ]

        self.batch_size = min(batch_size, self.n_samples - self.n_val) 
        if batch_size > self.batch_size :
            warning_string = ("The specified batch_size of " + str(batch_size) + " is larger than the available data set size of " + str(self.n_samples-self.n_val))
            warnings.warn(warning_string)

        validation_iteration        = 0
        self.validation_loss = np.zeros(int(ceil(epochs / val_stepwidth))+1)
        self.training_loss   = np.zeros(epochs)

        self.validation_loss_improving = np.zeros_like(self.validation_loss)
        self.validation_loss_improving *= np.nan

        self.batches_per_epoch = int(ceil(self.X_train.shape[1] / self.batch_size))
        
        start_time = time.time()

        for epoch in range(epochs) :    
            epoch_start_time   = time.time()
            epoch_loss = 0

            ## Let the magic begin with training on Mini Batches
            for batch in range(self.batches_per_epoch) :
                batch_start_time = time.time()
                x_batch, target_batch   = sklearn.utils.shuffle(self.X_train.T, self.target_train.T, n_samples=self.batch_size)
                
                y_batch = self.forward_pass(x_batch.T)
                self.backpropagation(y_batch.T, target_batch)
                batch_loss = self.cost_function(y_batch.T, target_batch, cost_function=self.cost_fct_name)

                ## Add L2 regularization to the loss
                reg_loss = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self.weights])) * 0.5 * self.lmbda / self.batch_size
                batch_loss += reg_loss
                epoch_loss += batch_loss

                self.optimizer()
            
            self.training_loss[epoch] = epoch_loss
            epoch_time                = time.time() - epoch_start_time
            #print("Epoch: {} of {}, took: {:.3f}s".format(epoch, epochs, epoch_time))

            ## Test against the validation set after each val_stepwidth
            if epoch % val_stepwidth == 0 or (epoch == epochs-1):
                y_validation = self.forward_pass(self.X_val)
                self.validation_loss[validation_iteration] = self.cost_function(y_validation.T, self.target_validation.T, cost_function=self.cost_fct_name)

                ## Add L2 regularization to the loss - make it more stable
                reg_loss = np.sum(np.array([np.dot(s.ravel(), s.ravel()) for s in self.weights])) * 0.5 * self.lmbda / self.batch_size
                self.validation_loss[validation_iteration] += reg_loss

                self.loss = self.validation_loss[validation_iteration]
                self.validation_loss_improving[validation_iteration] = self.best_loss

                if (self.best_loss is None) or (self.best_loss > self.validation_loss[validation_iteration]) :
                    self.best_loss  = self.validation_loss[validation_iteration]
                    self.best_param = [ params for params in (self.weights + self.biases) ]
                    self.bestEpoch  = epoch
                    self.validation_loss_improving[validation_iteration] = self.validation_loss[validation_iteration]
                    pickle.dump(self, open('nn.p', 'wb'))

                validation_iteration += 1

        self.validation_loss_improving[-1] = self.best_loss

        ## Set the weights and Biases to the best ones found (= lowest val error)
        self.weights = [w for w in self.best_param[:len(self.weights)]]
        self.biases  = [b for b in self.best_param[len(self.weights):]]
         

    def sgd(self) :
        '''
        This function does the stochastic gradient descent (SGD) and updates the weights and biases respectively
        '''
        ## do the update of the Weights and Biases
        for i, d_w in enumerate(self.d_weights) :
            self.weights[i] -= self.learning_rate * d_w

        for i, d_b in enumerate(self.d_biases) :
            self.biases[i]  -= self.learning_rate * d_b


    def initializeAdam(self) :
        '''
        This function Initialized the Adam optimizer which is State of the Art
        '''
        if not self.adam_initialized :
            self.t = 0
            self.learning_rate_init = self.learning_rate
            self.param = self.weights + self.biases
            
            ## First moment estimates
            self.m = [np.zeros_like(p) for p in self.param]
            # Corrected First moment estimates
            self.mh = [np.zeros_like(p) for p in self.param]
        
            ## Second moment estimates
            self.v = [np.zeros_like(p) for p in self.param]
            # Corrected second moment estimates
            self.vh = [np.zeros_like(p) for p in self.param]
            self.adam_initialized = True

    def adam(self) :
        '''
        This function implements Adam with the momentum terms
        along with the values proposed in the literature (https://arxiv.org/pdf/1412.6980.pdf)
        '''
        ## declare the parameters with the literature values
        beta1   = 0.9
        beta2   = 0.999
        epsilon = 1e-8
        self.t += 1
        t       = self.t

        ## Gradients
        self.grad = self.d_weights + self.d_biases

        ## m and v term 
        self.m = [beta1 * m + (1 - beta1) * g    for m,g in zip(self.m, self.grad)]

        self.v = [beta2 * v + (1 - beta2) * g**2 for v,g in zip(self.v, self.grad)]

        # adaptive learning rate
        self.learning_rate = self.learning_rate_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)

        change = [- self.learning_rate * m / (np.sqrt(v) + epsilon) for m,v in zip(self.m, self.v)]
        self.change = change

        ## Update step
        self.weights = [w + dw for w, dw in zip(self.weights, change[:len(self.weights)])]
        self.biases  = [b + db for b, db in zip(self.biases,  change[len(self.weights):])]

    def accuracy(self, X, y):
        '''
        this function calculates the accuracy score (sums correctly classified samples and divides it by length)
        INPUT: 
        -------
        X : numpy ndarray
            the sample(s) to be predicted in the network
        y: numpy.ndarray
            the corresponding ground truth labels

        OUTPUT: 
        ---------
            returns the accuracy score
        '''
        y_pred = self.predict(X.T) # feed forward pass
        y_pred = np.argmax(y_pred, axis=0) # get the higher probability label

        ## check for the predictions
        values, count = np.unique(y_pred, return_counts=True)
        if len(values) == 1: # only one label predicted
            print("There was always the label: {} predicted".format(values[0]))

        self.R2 = r2_score(y, y_pred)
        
        return np.sum(y.astype(int) == y_pred.astype(int)) / len(y)


#################### TESTING THIS GORGEOUS BEAUTY #################
from sklearn.model_selection import train_test_split
from LogisticRegression import get_data # my library

if __name__ == "__main__":
    # read the data
    filePath = os.path.join(os.path.join(os.getcwd(), 'data'), "default of credit card clients.xls")
    X, y = get_data(filePath, standardized=True, normalized=False)
    n_classes = 2
    epochs = 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2912)

    nn = NeuralNetwork(inputs           = X_train.shape[1],
                       outputs          = n_classes,
                       cost_function_name    = 'cross_entropy')
    nn.addLayer(activations = 'relu', neurons = 128)
    nn.addLayer(activations = 'relu', neurons = 128)
    nn.addLayer(activations = 'relu', neurons = 128)
    nn.addLayer(activations = 'relu', neurons = 64)
    nn.addLayer(activations = 'softmax', neurons = n_classes, output = True)
    
    nn.train_network(X_train.T,
           y_train.T,
           batch_size           = 64,
           learning_rate        = 0.0001,
           epochs               = epochs,
           val_size             = 0.2,
           val_stepwidth        = 10,
           optimizer            = 'adam',
           lmbda                = 0.0)

    print("Accuracy: {}".format(nn.accuracy(X_test, y_test)) )