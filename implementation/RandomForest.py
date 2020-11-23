import numpy as np
from implementation.DecisionTree import DecisionTree

class RandomForest:
    '''
    ### Parameters
    1. n_estimators : int , Default = 10\n
        Number of estimators
    2. impurity : {"entropy" , "gini"} : Default = "entropy"\n
        Metric for measuring impurity. Entropy and gini impurity
        respectively if "entropy" or "gini" for decision trees.
    '''

    def __init__(self, n_estimators = 10, impurity = "entropy"):
        self.n_estimators = n_estimators
        self.estimators = [DecisionTree() for _ in range(n_estimators)]
        self.impurity = impurity

    def train(self, x, y, threshold_impurity = 0.1, max_depth = 100,n_random_features=None,
            n_random_samples=None,):
        '''
        ### Parameters
        1. x : Numpy.array \n
            independent features
        2. y : Numpy.array \n
            actual classification
        3. threshold_impurity : float , Default = 0.1\n
            minimum impurity for splitting for individual trees
        4. max_depth : int , Default = 100\n
            maximum depth for splitting
        5. n_random_features : int , Default = None\n
            Uses 'n_random_features' features to 
            determine maximum information gain for individual trees. Uses all 
            if None.
        6. n_random_samples : int , Default = None\n
            Uses 'n_random_samples' samples to 
            determine maximum information gain for individual trees. Uses all 
            if None.  
        '''

        for estimator in range(self.n_estimators):
            self.estimators[estimator].fit(
                x,
                y,
                threshold_impurity = threshold_impurity,
                max_depth = max_depth,
                n_random_features = n_random_features,
                n_random_samples = n_random_samples,
                random_picking = True
                )
    
    def predict(self, x):
        '''
        ### Parameters
        1. x : Numpy.array\n
            independent features

        ### Returns 
            predicted classifications
        '''
        results = np.array(
            [self.estimators[estimator].predict(x) for estimator in range(self.n_estimators)]
            ).T

        predictions = []
        for i in results:
            values, counts = np.unique(i, return_counts = True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)