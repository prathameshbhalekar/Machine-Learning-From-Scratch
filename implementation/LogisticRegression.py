import numpy as np


class LogisticRegression:
    """
    ### Parameters
    1. regularization : { "None" , "L1" , "L2"} , Default = "None"\n
        Acts as Normal Linear Regression, Lasso Regression, Ridge
        Regression respectively if "None" , "L1" or "L2".  
    """

    def sigmoid(self, x):
        '''
        ### Parameters
            1. x : Numpy.array

        ### Returns 
            sigmoid of th array
        '''
        return 1 / (1 + np.exp(-x))

    def __init__(self, regulaization="None"):
        self.regulaization = regulaization

    def compute_cost(self, y, y_hat):
        '''
        ### Parameters
        1. y : Numpy.array\n
            actual classification
        2. y_hat : Numpy.array\n
            predicted classification

        ### Returns
            Log Loss of the model
        '''

        term1 = np.sum(np.multiply(y, np.log(1 - y_hat)))
        term2 = np.sum(np.multiply(1 - y, np.log(y_hat)))

        # Computing regularization term
        reg = 0.
        if(self.regulaization == "L1"):
            reg = self.lambda_ * sum(np.abs(self.weights))
        if(self.regulaization == "L2"):
            reg = sum((self.weights ** 2)) * self.lambda_

        return -(term1 + term2) / len(y) - reg

    def fit(self, x, y, print_every_nth_epoch=1, epochs=100, learning_rate=0.1, lambda_=0.01):
        '''
        ### Parameters
        1. x : Numpy.array\n
            independent features
        2. y : Numpy.array\n
            dependent features
        3. print_every_nth_epoch : Int , Default = 1\n
            prints mse after every 'print_every_nth_epoch'th epoch
        4. epochs : Int , Default = 100\n
            number of epoch for training the model
        5. learning_rate : Float , Default = 0.1\n
            hyper parameter for gradient descent algorithm.
            governs rate of diversion
        6. lambda_ : Float , Default = 0.01\n
            penalty factor for regulaization. Not 
            needed while using normal regression.

        '''

        assert x.shape[0] == y.shape[0], "unequal number of sizes"
        self.features = x.shape[1]
        self.lambda_ = lambda_
        n = x.shape[0]

        # Initializing weights and biases
        self.weights = np.random.rand(self.features)
        self.bias = 0.

        for epoch in range(epochs):
            y_hat = np.dot(x, self.weights) + self.bias
            y_hat = self.sigmoid(y_hat)
            diff = y - y_hat

            # Updating weights
            grad_w = + np.dot(x.T, diff) * learning_rate / n
            if(self.regulaization == "L1"):
                signs = np.where(self.weights > 0, 1, -1)
                grad_w = grad_w + signs * self.lambda_
            if(self.regulaization == "L2"):
                grad_w = grad_w + self.lambda_ * self.weights * 2
            self.weights += grad_w

            # Updating biases
            grad_b = + np.sum(diff) * learning_rate / n
            self.bias += grad_b

            if((epoch + 1) % print_every_nth_epoch == 0):
                print("--------- epoch {} -------> loss={} ----------"
                      .format((epoch + 1), self.compute_cost(y, y_hat)))

    def evaluate(self, x, y):
        '''
        ### Parameters
        1. x : Numpy.array\n
            independent features
        2. y : Numpy.array\n
            actual classification

        ### Returns
            Accuracy of the model
        '''

        pred = self.predict(x)
        pred = np.where(pred > 0.5, 1, 0)
        diff = np.abs(y - pred)
        return((len(diff) - sum(diff)) / len(diff))

    def predict(self, x):
        '''
        ### Parameters
        1. x : Numpy.array\n
            independent features

        ### Returns : Numpy
            Returns predicted classification
        '''
        return self.sigmoid(np.dot(x, self.weights) + self.bias)
