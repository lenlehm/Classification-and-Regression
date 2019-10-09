from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

def get_data(pathToFile, standardized=True, normalized=False):
    """
    reads in the Excel file and possibly standardizes or normalizes it
    INPUT: 
    ---------
    pathToFile: string
        string containing the path to the Excel file that should be used for the model
    standardized: boolean
        flag of whether the data should be standardized: (X - X.mean()) / X.std()
    normalized: boolean
        flag of whether the data should be normalized: (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    OUTPUT: 
    ----------
    X: pandas.DataFrame
        returns the dataframe with all the predictor variables (shape p x N || p = predictor variables, N = number of entries)
    y: pandas.DataFrame
        returns the targets for each of the training samples shape (1 x N)
    """
    nanDict = {}
    try:
        df = pd.read_excel(pathToFile, header=1, skiprows=0, index_col=0, na_values=nanDict)
    except Exception as e: 
        print("There is something wrong with the type of the file or the path: {}".format(e))
        df = pd.read_csv(pathToFile)

    ## TODO 
    ## DROP VALUES THAT ARE OUT OF RANGE!!
    # EDUCATION: [1,2,3,4] NO 5 and 6
    # martial status: [1,2,3] - NO 0 
    # df = df.drop(df[(df.BILL_AMT1 == 0) &
    #             (df.BILL_AMT2 == 0) &
    #             (df.BILL_AMT3 == 0) &

    # rename our target column
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
    target = 'defaultPaymentNextMonth'

    # get categorical and numeric columns
    numeric_colums = df.columns[df.dtypes == 'int64']
    categorical_columns = df.columns[df.dtypes == 'object']

    if len(categorical_columns) > 0:
        # One- hot encode the categorical values
        dummies = pd.get_dummies(df[categorical_columns], drop_first=True)
        df = pd.concat([dummies, df[numeric_colums]], axis=1)

    features = [i for i in df.columns if i != target]
    X = df[features]
    y = df[target]

    if standardized:
        X = (X - X.mean()) / X.std()
    if normalized: 
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # get the indices straight
    X.set_index(pd.Series(np.arange(len(df[features]))), inplace=True)
    return X, y

## Logistic Regression Class Implementation with Stochastic Gradient Descent
class LogisticRegression():
    def __init__(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)
        # Initialize parameter beta
        self.beta    = np.random.uniform(-0.5,0.5, self.X_train.shape[1])
        self.beta /= np.linalg.norm(self.beta)
        #self.beta = np.random.uniform(-0.5, 0.5, (self.X_train.shape[1], 1))


    def optimize(self, batch_size=64, epochs=128, eta='schedule', regularization=None, lamda=0.0, threshold=0.0001):
        """
        This method performs stochastic gradient descent to optimize its parameters

        INPUT: 
        --------
        batch_size: integer
            batchsize of the Stochastic Gradient Descent
        epochs: integer
            number of total iterations through the entire dataset (1 epoch = until every batch was propagated through network)
        eta: string
            learning rate for the optimizer ['schedule', None/ else]
        regularization: string
            type of regularization to be used ['l1', 'l2', else: No regularization]
        lamda: float
            regularization strength of the regularization method
        threshold: float
            threshold when we should stop optimizing, since there was no big difference
        
        OUTPUT:
        ----------
        Nothing - results and parameters stored in the member variables
        """

        X_train = self.X_train 
        X_test = self.X_test
        y_train = self.y_train 
        y_test = self.y_test

        self.batchSize = batch_size #int(X_train.shape[0] // m)
        iterPerEpoch = X_train.shape[0] // batch_size

        if eta == 'schedule':
            t0 = 5 ; t1 = 50
            learning_schedule = lambda t : t0 / (t + t1)
        else:
            learning_schedule = lambda t : eta

        if regularization == 'l2':
            reg_cost = lambda beta: lamda * np.sum(self.beta**2)
            reg_grad = lambda beta: 2 * lamda * self.beta
        elif regularization == 'l1':
            reg_cost = lambda beta: lamda * np.sum(np.abs(self.beta))
            reg_grad = lambda beta: lamda * np.sign(self.beta)
        elif regularization == None:
            reg_cost = lambda beta: 0
            reg_grad = lambda beta: 0

        #Stochastic Gradient Descent (SGD)
        for epoch in range(epochs + 1):
            # Shuffle training data
            # randomize = np.arange(X_train.shape[0])
            # X_train.set_index(pd.Series(np.arange(X_train.shape[0])), inplace=True)
            # np.random.shuffle(randomize)
            # X_train = X_train[randomize]
            # y_train = y_train[randomize]

            for i in range(iterPerEpoch):
                # create the batches
                rand_idx = np.random.randint(iterPerEpoch)
                xBatch = X_train[rand_idx * self.batchSize : (rand_idx+1) * self.batchSize]
                yBatch = y_train[rand_idx * self.batchSize : (rand_idx+1) * self.batchSize]

                y = np.dot(xBatch, self.beta) # somehow xBatch @ beta does not work ?!
                # logisitc probability 
                p = 1.0/(1 + np.exp(-y))
                eta = learning_schedule( (epoch*iterPerEpoch) + i)

                # update step for the parameter
                gradient   = -np.dot(xBatch.T, (yBatch - p)) / xBatch.shape[0] + reg_grad(self.beta) #scale + Lambda*beta/scale
                self.beta -= eta * gradient

            if (epoch % 10 == 0): # calculate accuracy after every 10th epoch
                logit_train = np.dot(X_train, self.beta)
                # binary cross entropy loss: 
                self.train_cost = 0.5*(-np.sum( (y_train * logit_train) - np.log(1 + np.exp(logit_train)) )) / X_train.shape[0] + reg_cost(self.beta)
                self.p_train = 1 / (1 + np.exp(-logit_train))
                # accuracy score
                self.train_accuracy.append( np.sum((self.p_train > 0.5) == y_train)/X_train.shape[0] )

                logit_test = np.dot(X_test, self.beta)
                # binary cross entropy on test data
                self.test_cost = 0.5*(-np.sum((y_test * logit_test) - np.log(1 + np.exp(logit_test))))/X_test.shape[0] + reg_cost(self.beta)
                self.p_test = 1/(1 + np.exp(-logit_test))
                # test accuracy score
                self.test_accuracy.append( np.sum((self.p_test > 0.5) == y_test) / X_test.shape[0] )

                print("Epoch: {} out of {} epochs".format(epoch, epochs))
                print("Cost during Training: {}, Test: {}".format(self.train_cost, self.test_cost))
                print("Accuracy in Training: {}, Test: {}".format(self.train_accuracy[i], self.test_accuracy[i]))
                print("--------------------------------------------------------------")

            # stopping criterion
            if(np.linalg.norm(gradient) < threshold):
                print("Gradient converged to given precision in {}. epoch and {}. iterations".format(epoch, i))
                break

# TEST THAT SHIT
if __name__ == '__main__':
    # read the dataset
    batch = 64
    cwd = os.getcwd()
    filename = "\\default of credit card clients.xls"
    filePath = cwd + filename
    X, y = get_data(filePath, standardized=False, normalized=False)
    logistic = LogisticRegression(X, y)

    logistic.optimize(batch_size=batch, regularization='l2', epochs=1000, lamda=10)

    # check for different lambdas
    # lambdas = np.logspace(-4,4,9)
    # for lamda in lambdas:
    #     logistic.optimize(batch_size=batch, regularization='l2', epochs=1000, lamda=lamda)