#Prasad Ostwal Scratchpad*

#importing necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from datetime import date,datetime
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.datasets import load_digits
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr,spearmanr
import my_funcs as func


# Load boston into Pandas Dataframe
data=load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names) 
boston['Price'] = data.target

# Train and fit data 
if True:
    y = boston['Price']
    x = boston.drop('Price',axis=1)
    
    #Split Train and Test set
    xtrain, xtest, ytrain,ytest= sklearn.model_selection.train_test_split(x,y,test_size=0.2)
    
    #Linear Regression
    lm= LinearRegression()
    lm.fit(xtrain,ytrain)
    ypred = lm.predict(xtest) 
    ypred2 = lm.predict(xtrain)
    
    #Ridge
    rr = Ridge(alpha=0.1)
    rr.fit(xtrain, ytrain)
    ypredRidge = rr.predict(xtest)
    
    #Lasso
    lasso= Lasso()
    lasso.fit(xtrain,ytrain)
    ypredLasso = lasso.predict(xtest)
    
    #PLotting Functions 
    func.my_scatter(ytest,ypred,"ypred-ytest",xname="ytest",yname="ypred",size=5,color='g')
    func.my_scatter(ytrain,ypred2,"ypred2-ytrain",xname="ytrain",yname="ypred2",size=5,color='b')
    func.my_scatter(ytest,ypredRidge,"ytest- YpredRidge",xname="ytest",yname="ypredRidge",size=5,color='y')
    func.my_scatter(ytest,ypredLasso,"ytest- YpredLasso",xname="ytest",yname="ypredLasso",size=5,color='m')
    
    #Printing Errors
    func.my_errors(ytest,ypred,title="Ytest -  Ypred")
    func.my_errors(ytrain,ypred2,title="Ytrain - Ypred")
    func.my_errors(ytest,ypredRidge,title="Ytrain - YpredRidge")
    func.my_errors(ytest,ypredLasso,title="Ytrain - YpredLasso")
    
    

    
    
