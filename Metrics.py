import numpy as np
import matplotlib.pyplot as plt

def accuracy (y, y_hat):
    '''
    Function to calculate accuracy
    :param y : actual values 
    :param y_hat : predicted values
    :return : accuracy 
    '''
    
    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    correct = 0

    for (actual, predicted) in zip(y, y_hat):
        if(actual == predicted):
            correct += 1
    
    return correct / len(y)

def __true_positive (y, y_hat, positive_value = 1):
    '''
    Function to calculate number of true positives
    :param y : actual values 
    :param y_hat : predicted values
    :param positive_value : value of positive sample, Default = 1
    :return : true positives 
    '''

    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    count = 0

    for (actual, predicted) in zip(y, y_hat):
        if(actual == positive_value and predicted == positive_value):
            count += 1
    
    return count


def __true_negetive (y, y_hat, positive_value = 1):
    '''
    Function to calculate number of true negetives
    :param y : actual values 
    :param y_hat : predicted values
    :param positive_value : value of positive sample, Default = 1
    :return : true negetives
    '''

    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    count = 0

    for (actual, predicted) in zip(y, y_hat):
        if(actual != positive_value and predicted != positive_value):
            count += 1
    
    return count


def __false_positive (y, y_hat, positive_value = 1):
    '''
    Function to calculate number of false positives
    :param y : actual values 
    :param y_hat : predicted values
    :param positive_value : value of positive sample, Default = 1
    :return : false positives 
    '''

    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    count = 0

    for (actual, predicted) in zip(y, y_hat):
        if(actual != positive_value and predicted == positive_value):
            count += 1
    
    return count


def __false_negetives (y, y_hat, positive_value = 1):
    '''
    Function to calculate number of false negetives
    :param y : actual values 
    :param y_hat : predicted values
    :param positive_value : value of positive sample, Default = 1
    :return : false negetives
    '''

    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    correct = 0

    for (actual, predicted) in zip(y, y_hat):
        if(actual == positive_value and predicted != positive_value):
            correct += 1
    
    return correct


def precision ( y, y_hat, positive_value = 1):
    '''
    Function to calculate number of true positives
    :param y : actual values 
    :param y_hat : predicted values
    :param positive_value : value of positive sample, Default = 1
    :return : true positives 
    '''

    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    true_positive = __true_positive(y, y_hat, positive_value = positive_value)
    false_positive = __false_positive(y, y_hat, positive_value = positive_value)

    precision = true_positive / (true_positive + false_positive)

    return precision

def recall ( y, y_hat, positive_value = 1):
    '''
    Function to calculate number of true positives
    :param y : actual values 
    :param y_hat : predicted values
    :param positive_value : value of positive sample, Default = 1
    :return : true positives 
    '''

    assert len(y) == len(y_hat) , "Unequal length of y and y_hat"

    true_positive = __true_positive(y, y_hat, positive_value = positive_value)
    false_negetives = __false_negetives(y, y_hat, positive_value = positive_value)

    recall = true_positive / (true_positive + false_negetives)

    return recall


def f1(y, y_hat, positive_value = 1):
    """
    Function to calculate f1 score
    :param y: list of true values
    :param y_hat: list of predicted values
    :param positive_value : value of positive sample, Default = 1
    :return: f1 score
    """
    p = precision(y, y_hat, positive_value= positive_value)
    r = recall(y, y_hat, positive_value= positive_value)
    score = 2 * p * r / (p + r)
    return score

def tpr(y, y_hat, positive_value = 1):
    """
    Function to calculate tpr
    :param y: list of true values
    :param y_hat: list of predicted values
    :param positive_value : value of positive sample, Default = 1
    :return: tpr/recall
    """

    return recall(y, y_hat, positive_value= positive_value)

def fpr(y, y_hat, positive_value = 1):
    """
    Function to calculate tpr
    :param y: list of true values
    :param y_hat: list of predicted values
    :param positive_value : value of positive sample, Default = 1
    :return: tpr/recall
    """

    fp = __false_positive(y, y_hat, positive_value= positive_value)
    tn = __true_negetive(y, y_hat, positive_value= positive_value)
    return fp / (tn + fp)

def roc(y, y_pred, thresholds, positive_value = 1):
    """
    Function to plot roc curve
    :param y: list of true values
    :param y_pred: list of predicted probablities
    :param thresholds: list of threshold
    :param positive_value : value of positive sample, Default = 1
    """
    tpr_list = []
    fpr_list= []
    for threshold in thresholds:
        y_hat = np.where(y_pred >= threshold, positive_value, 0)
        
        tpr_val = tpr(y, y_hat, positive_value = positive_value)
        fpr_val = fpr(y, y_hat, positive_value = positive_value)

        tpr_list.append(tpr_val)
        fpr_list.append(fpr_val)

    plt.plot( fpr_list, tpr_list,)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("ROC Curve")
    plt.show()

    

























