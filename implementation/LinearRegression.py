import numpy as np


class LinearRegression:
    """
    ### Parameters
    1. regularization : { "None" , "L1" , "L2"} , Default = "None"
        Acts as Normal Linear Regression, Lasso Regression, Ridge
        Regression respectively if "None" , "L1" or "L2".  
    """

    def __init__(self, regularization="None"):
        self.regularization = regularization

    def mse(self, y, y_hat) -> float:
        '''
        ### Parameters
        1. y : Numpy.array
            Actual dependent feature
        2. y_hat : Numpy.array
            Predicted dependent feature

        ### Returns : Float
            The mean squared error of the model.
        '''
        diff = y - y_hat
        reg = 0.

        # Computing regularization term
        if(self.regularization == "L1"):
            reg = np.sum(np.abs(self.weights)) * self.lambda_
        if(self.regularization == "L2"):
            reg = np.sum(self.weights ** 2) * self.lambda_

        return np.sum(diff ** 2) / (2 * len(y)) + reg

    def fit(self, x, y, print_every_nth_epoch=1, epochs=100, learning_rate=0.1, lambda_=0.01):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features
        2. y : Numpy.array
            dependent features
        3. print_every_nth_epoch : Int , Default = 1
            prints mse after every 'print_every_nth_epoch'th epoch
        4. epochs : Int , Default = 100
            number of epoch for training the model
        5. learning_rate : Float , Default = 0.1
            hyper parameter for gradient descent algorithm.
            governs rate of diversion
        6. lambda_ : Float , Default = 0.01
            penalty factor for regulaization. Not 
            needed while using normal regression.

        '''

        # Check for shape of x and y
        assert y.shape[0] == x.shape[0], "Number of entries don't match"

        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.features = x.shape[1]
        n = x.shape[0]

        # Initializing weights and biases to 0
        self.weights = np.zeros(self.features)
        self.bias = 0

        for epoch in range(epochs):
            y_bar = np.dot(x, self.weights) + self.bias
            diff = y - y_bar

            # Updating weights
            grad_w = np.dot(x.T, diff)*self.learning_rate/n

            if(self.regularization == "L1"):
                sign = np.where(self.weights > 0, 1, -1)
                grad_w += sign * self.lambda_
            if(self.regularization == "L2"):
                grad_w += self.lambda_ * 2 * self.weights

            self.weights += grad_w

            # Updating bias
            grad_b = np.sum(diff) * self.learning_rate / n
            self.bias += grad_b

            if((epoch + 1) % print_every_nth_epoch == 0):
                print("--------- epoch {} -------> loss={} ----------"
                      .format((epoch + 1), self.mse(y, y_bar)))

    def predict(self, x):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features

        ### Returns : Numpy
            Returns predicted dependent feature
        '''

        assert self.features == x.shape[1], "Number of features don't match"
        return np.dot(x, self.weights) + self.bias
