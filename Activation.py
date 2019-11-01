import numpy as np
import sys

class Activation :
    '''
    class to make a modular approach for the activation functions along with their derivatives
    Implemented Activation Functions: 
        - Softmax
        - ReLU
        - Leaky ReLU
        - ELU
        - Sigmoid
        - tanh
        - Identity

    '''
    def __init__(self, function=None, alpha=None):
        self.derivative = None
        self.function = function if (function is not None) else 'sigmoid'
        self.alpha    = alpha    if (alpha    is not None) else 0.01
        self.name     = 'sigmoid'
        self.function = self._parseFunctionString(self.function)

    def set(self, function=None, alpha=None):
        '''
        constructor method to create the member variables and init them properly
        '''
        self.alpha = alpha if (alpha is not None) else self.alpha
        if function is not None :
            self.function = function
            self.name     = function
            self.function = self._parseFunctionString(self.function)
    

    def _parseFunctionString(self, string) :
        '''
        check which activation function to use and call the mathematical function
        '''
        self.name = string
        if string == 'sigmoid' :
            self.derivative = self._sigmoid_derivative
            return self._sigmoid
        elif string == 'tanh' : 
            self.derivative = self._tanh_derivative
            return self._tanh
        elif string == 'relu' :
            self.derivative = self._relu_derivative
            return self._relu
        elif string == 'leakyrelu' or string == 'leaky_relu' :
            self.derivative = self._leakyrelu_derivative
            return self._leakyrelu
        elif string == 'identity' :
            self.derivative = self._identity_derivative
            return self._identity
        elif string == 'elu' :
            self.derivative = self._elu_derivative
            return self._elu
        elif string == 'softmax' :
            self.derivative = self._softmax_derivative
            return self._softmax
        else :
            raise ValueError("Unrecognized activation function <" + str(string) + ">.")

    ## --------------------- DOWN THERE IS THE MATHEMATICAL TURD OF EACH OF THE FUNCTIONS ----------------
    def _sigmoid(self, x) :
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_derivative(self, x) :
        return x * (1.0 - x)

    def _tanh(self, x) :
        return np.tanh(x)

    def _tanh_derivative(self, x) :
        return 1.0 - x**2

    def _relu(self, x) :
        return np.maximum(0, x)

    def _relu_derivative(self, x) :
        return np.maximum(x, 0.0)

    def _leakyrelu(self, x) :
        return (x >= 0.0) * x + (x < 0.0) * (self.alpha * x)

    def _leakyrelu_derivative(self, x) :
        dx = np.ones_like(x)
        dx[x < 0] = 0.01
        return dx

    def _identity(self, x) :
        return x

    def _identity_derivative(self, x) :
        return 1.0

    def _elu(self, x) :
        neg = x < 0.0
        x[neg] = (np.exp(x[neg]) - 1.0)
        return x

    def _elu_derivative(self, x) :
        neg = x < 0.0
        x[neg] = np.exp(x[neg])
        x[x >= 0.0] = 1.0

    def _softmax(self, x) :
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def _softmax_derivative(self, x) :
        return x

    def __call__(self, x) :
        return self.function(x)
