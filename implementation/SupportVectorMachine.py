import numpy as np


class SupportVectorMachine:
    '''
    ### Parameters
    1. C : float , Default = 10.
        Hyperparameter for loss function
    2. kernel : {"None" , "gaussian"} , Default = "None"
        Uses no kernel , gaussian kernel if "None" or
        "gaussian" respectively
    3. sigma_sq : float , Default = 0.1
        Hyperparameter for gaussian kernel
    '''

    def __init__(self, C = 10., kernel = "None", sigma_sq = 0.1):
        self.C = C
        self.sigma_sq = sigma_sq
        self.kernel = kernel

    def __similarity(self, x, l):
        '''
        ### Parameters
        1. x : Numpy.array
            first vector
        2. l : Numpy.array
            second vector
        
        ### Returns : float
            similarity between "x" and "l"
        '''
        return np.exp(-sum((x-l) ** 2) / (2 * self.sigma_sq))

    def gaussian_kernel(self, x1, x):
        '''
        ### Parameters 
        1. x1 : Numpy.array
            vector on which kernel is to be applied
        2. x : Numpy.array
            vector w.r.t which kernel is to be applied
        
        ### Returns
            Kernelized for of "x1" 
        '''
        m = x.shape[0]
        n = x1.shape[0]
        op = [[self.__similarity(x1[x_index], x[l_index])
               for l_index in range(m)] for x_index in range(n)]
        return np.array(op)

    def loss_function(self, y, y_hat):
        '''
        ### Parameters
        1. y : Numpy.array
            actual classification
        2. y_hat : Numpy.array
            predicted classification
        
        ### Returns 
            hinge loss of the model
        '''
        
        sum_terms = 1 - y * y_hat
        sum_terms = np.where(sum_terms < 0, 0, sum_terms)
        
        return (self.C * np.sum(sum_terms) / len(y) + sum(self.weights ** 2) / 2)

    def fit(self, x_train, y_train, print_every_nth_epoch = 1, epochs=100, learning_rate=0.1):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features
        2. y : Numpy.array
            actual classification
        3. print_every_nth_epoch : Int , Default = 1
            prints loss after every 'print_every_nth_epoch'th epoch
        4. epochs : Int , Default = 100
            number of epoch for training the model
        5. learning_rate : Float , Default = 0.1
            hyper parameter for gradient descent algorithm.
            governs rate of diversion
        '''
        y = y_train.copy()
        x = x_train.copy()
        self.features = x.shape[1]
        self.initial = x.copy()
        n = x.shape[0]

        self.weights = np.zeros(self.features)
        self.bias = 0.

        assert x.shape[0] == y.shape[0], "Samples of x and y don't match."

        # Appling kernel
        if(self.kernel == "gaussian"):
            x = self.gaussian_kernel(x, x)
            m = x.shape[0]
            self.weights = np.zeros(m)

        for epoch in range(epochs):
            y_hat = np.dot(x, self.weights) + self.bias

            # Updating weights
            grad_weights = (-self.C * np.multiply(y, x.T).T + self.weights).T
            
            for weight in range(self.weights.shape[0]):
                grad_weights[weight] = np.where(
                    1 - y_hat <= 0, self.weights[weight], grad_weights[weight]
                    )
            
            grad_weights = np.sum(grad_weights, axis = 1)
            self.weights -= learning_rate * grad_weights / n
            
            # Updating Bias
            grad_bias = -y * self.bias
            grad_bias = np.where(1-y_hat <= 0, 0, grad_bias)
            grad_bias = sum(grad_bias)
            self.bias -= grad_bias * learning_rate / n
            
            if((epoch + 1) % print_every_nth_epoch == 0):
                print("--------------- Epoch {} --> Loss = {} ---------------"
                    .format(epoch+1, self.loss_function(y, y_hat)))

    def evaluate(self, x, y):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features
        2. y : Numpy.array
            actual classification

        ### Returns
            Accuracy of the model
        '''
        pred = self.predict(x)
        pred = np.where(pred == -1, 0, 1)
        diff = np.abs(np.where(y == -1, 0, 1)-pred)
        return((len(diff)-sum(diff))/len(diff))

    def predict(self, x):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features

        ### Returns : Numpy
            Returns predicted classification
        '''
        # Applying kernel
        if(self.kernel == "gaussian"):
            x = self.gaussian_kernel(x, self.initial)
        
        return np.where(np.dot(x, self.weights)+self.bias > 0, 1, -1)
