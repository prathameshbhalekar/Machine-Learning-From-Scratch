import random 
import pandas as pd
import numpy as np
import random
import math

class DecisionTreeRegression:
    
    class node:
        '''
        Individual node for decision tree.
        '''

        def __init__(self, isleaf=False):
            self.isleaf = isleaf
            self.left = None
            self.right = None
            self.value = None
            self.feature = None
    
    def __fit_util(self, df : pd.DataFrame, depth : int) -> node:
        
        '''
        ### Parameters
        1. df : Pandas.Dataframe
            dataframe to be classified
        2. depth : int
            height of tree above the node

        ### Returns : node
            subtree to classify "df"
        '''

        # Check if threshold std. deviation o max depth is reached
        if(
            math.sqrt(df['op'].var()) <= self.threshold_impurity or 
            depth >= self.max_depth or
            df.shape[0] < self.min_samples ):
            n = self.node(isleaf= True)
            n.value = df['op'].mean()
            return n

        min_var = float("inf")
        min_feature = None
        min_value = None

        features = list(df.columns)
        features.remove('op')

        # Picking random features
        if(self.n_random_features != None):
            features = random.sample(features, self.n_random_features)

        for feature in features:

            values = list(df[feature].unique())
            
            # Picking random values
            if(self.n_random_samples != None):
                values = random.sample(values, self.n_random_samples)

            for value in values:
                df1 = df[df[feature] <= value]
                df2 = df[df[feature] > value]
                # Checking if any dataframe is empty
                if(df1.shape[0] == 0 or df2.shape[0]==0 ):
                    continue

                if(min(df1['op'].var(),df2['op'].var()) < min_var):
                    min_var = min(df1['op'].var(),df2['op'].var())
                    min_feature = feature
                    min_value = value
        
        if(min_value == None):
            print(df)
        df1 = df[df[min_feature] <= min_value]
        df2 = df[df[min_feature] > min_value]

        n = self.node()
        n.feature = min_feature
        n.value = min_value
        
        # check if left is leaf or right
        if(df1['op'].var() < df2['op'].var()):
            left = self.node(isleaf= True)
            left.value = df1['op'].mean()

            n.left = left
            n.right = self.__fit_util(df2, depth + 1)       
        else :
            right = self.node(isleaf= True)
            right.value = df2['op'].mean()

            n.right = right
            n.left = self.__fit_util(df1, depth + 1)
        
        return n


    def fit (
        self,
        x : np.array, 
        y : np.array, 
        threshold_impurity : float = 0.1,
        max_depth : int = 100,
        n_random_features : int = None, 
        n_random_samples : int = None,
        min_samples : int = 5
        ):

        '''
        ### Parameters
        1. x : Numpy.array 
            independent features
        2. y : Numpy.array 
            actual classification
        3. threshold_impurity : float , Default = 0.1
            minimum impurity for splitting
        4. max_depth : int , Default = 100
            maximum depth for splitting
        5. n_random_features : int , Default = None
            Uses 'n_random_features' features to 
            determine maximum information gain. Uses all 
            if None.
        6. n_random_samples : int , Default = None
            Uses 'n_random_samples' samples to 
            determine maximum information gain. Uses all 
            if None.
        7. min_samples : int , Default = 5
            Minimum samples required for splitting 
        '''

        assert x.shape[0] == y.shape[0] , "Unequal lengths of x and y"

        self.threshold_impurity = threshold_impurity
        self.max_depth = max_depth
        self.n_random_features = n_random_features
        self.n_random_samples = n_random_samples
        self.min_samples = min_samples

        df = pd.DataFrame(x)
        df['op'] = y

        self.root = self.__fit_util(df, 0)
    
    def __predict_util(self, n : node, df : pd.DataFrame) :
        '''
        ### Parameters
        1. n : node
            Current node
        2. df : Pandas.DataSeries

        ### Returns : Any
            Predicited Classification
        '''

        if(n.isleaf):
            return n.value

        if(df[n.feature] <= n.value):
            return self.__predict_util(n.left, df)
        else:
            return self.__predict_util(n.right, df)
    
    def predict(self, x):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features

        ### Returns 
            predicted dependent features
        '''
        
        df = pd.DataFrame(x)
        n = df.shape[0]

        y = [self.__predict_util(self.root, df.iloc[i, :]) for i in range(n)]
            
        return np.array(y)

