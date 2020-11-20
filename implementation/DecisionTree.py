import numpy as np
import pandas as pd
import random


class DecisionTree:
    '''
    ### Parameters
    1. impurity : { "entropy", "gini" } , Default = "entropy"
        Metric for measuring impurity. Entropy and gini impurity
        respectively if "entropy" or "gini"
    '''

    def __init__(self, impurity="entropy"):
        self.impurity = impurity

    class node:
        '''
        Individual node for decision tree.
        '''

        def __init__(self, isleaf=False):
            self.isleaf = isleaf

    def __entropy(self, df) -> float:
        '''
        ### Parameters
        1. df : Pandas.Dataframe
            Dataframe at particular node.

        ### Returns : float
            returns entropy of the dataframe
        '''

        sum = 0.
        n = df.shape[0]

        for i in df['o/p'].value_counts():
            pi = i / n
            sum -= pi * np.log(pi)

        return sum

    def __gini_impurity(self, df) -> float:
        '''
        ### Parameters
        1. df : Pandas.Dataframe
            Dataframe at particular node.

        ### Returns : float
            returns gini impurity of the dataframe
        '''
        sum = 0.
        n = df.shape[0]

        for i in df['o/p'].value_counts():
            sum += (i / n) ** 2

        return 1 - sum

    def __information_gain(self, df, df1, df2) -> float:
        '''
        ### Parameters
        1. df : Pandas.Dataframe
            Initial dataframe
        2. df1 : Pandas.Dataframe
            First dataframe after split
        3. df2 : Pandas.Dataframe
            Second dataframe after split

        ### Returns : float
            Information Gain after split
        '''

        # To prevent spliting if one of the dataframe is empty
        if(df1.shape[0] == 0 or df2.shape[0] == 0):
            return float("-inf")

        s = df.shape[0]
        sv1 = df1.shape[0]
        sv2 = df2.shape[0]

        h = 0.
        h1 = 0.
        h2 = 0.

        if(self.impurity == 'entropy'):
            h = self.__entropy(df)
            h1 = self.__entropy(df1)
            h2 = self.__entropy(df2)
        else:
            h = self.__gini_impurity(df)
            h1 = self.__gini_impurity(df1)
            h2 = self.__gini_impurity(df1)

        ig = h - sv1 * h1 / s - sv2 * h2 / s

        return ig

    def __fit_util(self, df, depth) -> node:
        '''
        ### Parameters
        1. df : Pandas.Dataframe
            dataframe to be classified
        2. depth : int
            height of tree above the node

        ### Returns : node
            subtree to classify "df"
        '''

        # Check for impurity or depth threshold
        impurity = 0
        if(self.impurity == 'entropy'):
            impurity = self.__entropy(df)
        else:
            impurity = self.__gini_impurity(df)

        if(impurity <= self.threshold_impurity or depth == self.max_depth):
            n = self.node(isleaf=True)
            n.classification = df['o/p'].mode()[0]
            return n

        max_ig = float('-inf')
        max_feature = df.columns[0]
        max_division = 0

        features = list(df.columns)
        features.remove('o/p')
        # Pick up feature and it dividing value for maximum
        # information gain
        for feature in random.sample(features, self.n_random_features):
            unique_samples = list(df[feature].unique())

            if(self.n_random_samples != None):
                unique_samples = random.sample(
                    unique_samples,
                    min(self.n_random_samples, len(unique_samples))
                )

            for division in unique_samples:
                df1 = df[df[feature] <= division]
                df2 = df[df[feature] > division]
                ig = self.__information_gain(df, df1, df2)
                if(ig > max_ig):
                    max_ig = ig
                    max_feature = feature
                    max_division = division

        df1 = df[df[max_feature] <= max_division]
        df2 = df[df[max_feature] > max_division]

        new_node = self.node()
        new_node.feature = max_feature
        new_node.division = max_division

        # Constructing left and right subtree
        new_node.left = self.__fit_util(df1, depth + 1)
        new_node.right = self.__fit_util(df2, depth + 1)

        return new_node

    def fit(
            self,
            x,
            y,
            threshold_impurity=0.1,
            max_depth=100,
            n_random_features=None,
            n_random_samples=None,
            random_picking=False):
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
        7. random_picking : boolean , Default = False
            Picks random samples with replacement for
            training if True(for Random Forest).
        '''

        assert x.shape[0] == y.shape[0], "Unequal length of Dataframes"

        if(n_random_features == None):
            self.n_random_features = x.shape[1]
        else:
            self.n_random_features = n_random_features

        df = pd.DataFrame(x)
        df['o/p'] = pd.DataFrame(y)

        self.n = x.shape[0]
        self.n_random_samples = n_random_samples
        self.max_depth = max_depth
        self.threshold_impurity = threshold_impurity
        self.x = x
        self.y = y

        if(random_picking):
            df = df.sample(self.n, replace=True, axis=0)
        self.tree = self.__fit_util(df, 0)

    def __predict_util(self, n, df):
        '''
        ### Parameters
        1. n : node
            Current node
        2. df : Pandas.DataSeries

        ### Returns : Any
            Predicited Classification
        '''

        if(n.isleaf):
            return n.classification

        if(df[n.feature] <= n.division):
            return self.__predict_util(n.left, df)
        else:
            return self.__predict_util(n.right, df)

    def predict(self, x):
        '''
        ### Parameters
        1. x : Numpy.array
            independent features

        ### Returns 
            predicted classifications
        '''
        df = pd.DataFrame(x)
        n = df.shape[0]
        y = []
        for i in range(n):
            y.append(self.__predict_util(self.tree, df.iloc[i, :]))
        return np.array(y)
