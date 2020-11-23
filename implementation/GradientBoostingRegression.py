import numpy as np
import numpy
from implementation.DecisionTreeRegression import DecisionTreeRegression

class GradientBoostingRegression:
    '''
        ### Parameters 
        1. estimator_count : int , Default = 100 \n    
            number of estimators 
    '''
    def __init__(self, estimator_count : int = 100) -> None:
        self.estimator_count = estimator_count

    def fit (
        self,
        x : np.array, 
        y : np.array, 
        learning_rate : float = 0.1,
        threshold_impurity : float = 0.1,
        max_depth : int = 100,
        n_random_features : int = None, 
        n_random_samples : int = None,
        min_samples : int = 5,
        print_every_nth_estimator = 1
        ):
        '''
        1. x : Numpy.array \n 
            independent features
        2. y : Numpy.array \n
            actual dependent features
        3. learning_rate : float , Default = 0.1 \n
            say for each estimator
        4. threshold_impurity : float , Default = 0.1\n
            minimum impurity for splitting for individual tree
        5. max_depth : int , Default = 100\n
            maximum depth for splitting for individual tree
        6. n_random_features : int , Default = None\n
            Uses 'n_random_features' features to 
            determine maximum information gain individual tree. Uses all 
            if None.
        7. n_random_samples : int , Default = None\n
            Uses 'n_random_samples' samples to 
            determine maximum information gain for individual tree. Uses all 
            if None.
        8. min_samples : int , Default = 5\n
            Minimum samples required for splitting individual tree
        9. print_every_nth_estimator : int , Default = 1\n
            Notifies after training "print_every_nth_estimator"th estimator.
        '''
    
        assert x.shape[0] == y.shape[0] , "Unqual shape of x and y"

        self.learning_rate = learning_rate
        n = x.shape[0]

        self.base_value = np.mean(y)

        last_predictions = np.full(n, self.base_value)
        residuals = y - last_predictions
        self.estimators = []

        for estimator in range(self.estimator_count):
            
            tree = DecisionTreeRegression()
            tree.fit(x, residuals, threshold_impurity, 
                max_depth, n_random_features, n_random_samples, min_samples)
            
            last_predictions += self.learning_rate * tree.predict(x)
            residuals = y - last_predictions
            self.estimators.append(tree)
            
            if((estimator + 1) % print_every_nth_estimator == 0):
                print("Completed training {}th estimator".format(estimator + 1))

    def predict (self, x : numpy.array):
        '''
        ### Parameters
        1. x : Numpy.array\n
            Independent features
        '''
        
        n = x.shape[0]
        results = np.full(n, self.base_value)
        
        for estimator in self.estimators:
            predictions = estimator.predict(x)
            results += self.learning_rate * predictions
        
        return results


    
