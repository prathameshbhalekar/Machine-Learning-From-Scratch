from functools import total_ordering
import numpy as np
from implementation.DecisionTree import DecisionTree
# using SAMME algorithm 
class AdaBoost:
    '''
    ### Parameters
    1. n_estimators : int , Default = 100
        Number of estimators
    2. random_sampling : boolean , Default = False
        If True drops samples whith low weights after training each
        estimators
    3. impurity : { "entropy", "gini" } , Default = "entropy"
        Metric for measuring impurity. Entropy and gini impurity
        respectively if "entropy" or "gini"
    '''

    def __init__(self, n_estimators = 100, random_sampling = False, impurity = "entropy"):
        self.n_estimators = n_estimators
        self.random_sampling = random_sampling
        self.impurity = impurity

        self.TOTAL_ERROR_ERROR_TERM=0.00001
    
    def __get_total_sample_weights(self,predictions,y,sample_weights) -> float:
        '''
        ### Parameters
        1. predictions : Numpy.array
            Predicted classification
        2. y : Numpy.array
            Actual classification
        3. sample_weights : Numpy.array
            Weights of samples
        
        ### Returns 
            Total weights of incorrectly classified samples
        
        '''

        # Calculating weights of incorrectly classified samples
        total_sample_weights = 0.0
        for (prediction, actual, sample_weight) in zip(predictions, y, sample_weights):
            if(prediction != actual):
                total_sample_weights += sample_weight
        return total_sample_weights

    def __get_say(self,total_sample_weight):
        '''
        ### Parameters 
        1. total_sample_weight : float
            Total weights of incorrectly classified samples
        '''
        return np.log((1 - total_sample_weight) / total_sample_weight) / 2

    def fit(self, x, y, print_every_nth_estimator = 1):
        '''
        ### Parameters
        1. x : Numpy.array
            Independent features
        2. y : Numpy.array
            Actual classification
        3. print_every_nth_estimator : int , Default = 1
            Notifies after training "print_every_nth_estimator"th estimator
        '''

        assert x.shape[0] == y.shape[0] , "Unequal sample size"
        n = x.shape[0]
        features = x.shape[1]

        sample_weights = np.full(n, 1 / n)
        self.estimators = []
        self.estimator_say = []
        
        for estimator_count in range(self.n_estimators):
            
            # Picking feature with minimum total sample weight
            min_estimator = None
            min_weight = np.float('inf')
            for feature in range(features):
                estimator = DecisionTree(self.impurity)
                data = np.array([x[:,feature]]).T
                estimator.fit(data, y, max_depth = 2, n_random_samples = 10)
                predictions = estimator.predict(data)
                
                total_sample_weights = self.__get_total_sample_weights(predictions, y, sample_weights)

                if(total_sample_weights < min_weight):
                    min_estimator = estimator
                    min_weight = total_sample_weights

            self.estimators.append(min_estimator)
            # Error term for calculating say in case total_sample_weight is zero
            min_weight += self.TOTAL_ERROR_ERROR_TERM
            
            say = self.__get_say(min_weight)
            self.estimator_say.append(say)
            
            # Standardizing sample_weights
            predictions = min_estimator.predict(x)
            sample_weights = np.where(
                y != predictions,
                sample_weights * np.exp(np.abs(say)),
                sample_weights * np.exp(-np.abs(say)))
            sample_weights = sample_weights / np.sum(sample_weights)

            # Removing samples with low weights
            if(self.random_sampling):
                cumsum = np.cumsum(sample_weights)
                random_nums = np.random.rand(len(sample_weights))
                positions = [np.min(np.where(cumsum >= value )) for value in random_nums]
                x = x[positions]
            
            if((estimator_count + 1) % print_every_nth_estimator == 0):
                print("Completed training estimator {}".format(estimator_count + 1))

    def predict(self,x):
        '''
        ### Parameters
        1.x : Numpy.array 
            Independent Features

        ### Returns : Numpy.array
            Predicted Categories
        '''

        estimators_results = np.array([estimator.predict(x) for estimator in self.estimators]).T
        result = []

        for predictions in estimators_results:
            
            total_weights = {}
            
            for i in range(self.n_estimators):
                prediction = predictions[i]
                if prediction not in total_weights.keys():
                    total_weights[prediction] = 0
                total_weights[prediction] += self.estimator_say[i]
            
            result.append(max(total_weights, key = total_weights.get))
        
        return np.array(result)
    