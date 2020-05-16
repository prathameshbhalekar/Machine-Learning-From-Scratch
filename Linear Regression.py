#   Linear regression is used to predict and in some cases classify linear data. It works by defining a best fit line which is the
#   line such that error from all points is minimum. This line is called best fit line. 

#   We can predict new data by substituting features in the best fit line

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from statistics import mean
from matplotlib import style
import random
style.use('ggplot')

#    This function is used to get the slope of the best fit line
def get_slope(Xs,Ys):
    slope=(mean(Xs)*mean(Ys)-(mean(Xs*Ys)))/(mean(Xs)**2-mean(Xs**2))
    return slope
    
#    This function is used to get the y intercept of the best fit line    
def get_y_intercept(Xs,Ys,slope):
    c=np.mean(Ys)-slope*np.mean(Xs)
    return c
    
#    This function gives us linear data with some variance to emulate real world data    
def get_data(n,variance,step,linear=True):
    Y=[]
    X=[]
    count=1
    for i in range (0,n):
        X.append(i)
        Y.append(count+random.randrange(-variance,variance))
        if(linear and linear=='pos'):
            count+=step
        elif(linear and linear=='neg'):
            count-=step
    return np.array(X),np.array(Y)
    
#    squared error is the square of difference between actual data and in this case points on the line passing through mean
#    data and the best fit line

#    It is not necessary to use square of the error. You may use any even power of the error but square is the standard.
def squared_error(ys_original,ys_line):
    return sum((ys_line-ys_original)**2)

#    coefficient of determination is one of many ways to determine how good is the best fit line.
#    it is calculated by compairing squared error of best fit line with with that of line passing through mean of all data

def coefficient_of_determination(ys_original,ys_line):
    mean_array=[mean(ys_original) for _ in ys_original]
    mean_line=squared_error(ys_original,mean_array)
    best_fit_line=squared_error(ys_original,ys_line)
    return 1-(best_fit_line/mean_line)
    
#   This function predicts the values using pluging features into best fit line.

def predict(slope,c,x):
    return slope*x+c

X,Y=get_data(40,10,2,linear='pos')
for i in range (0,len(X)):
    plt.scatter(X[i],Y[i],color='b')

slope=get_slope(X,Y)
c=get_y_intercept(X,Y,slope)
Ys=[]
for i in X:
    Ys.append(slope*i+c)
plt.plot(X,Ys)
print(get_coefficent_of_determination(Y,Ys))    
x=25
y=predict(slope,c,x)
plt.scatter(x,y,s=100,color='g')
print(y)
plt.show()
