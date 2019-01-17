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
import my_funcs

# Load boston into Pandas Dataframe
data=load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names) 
boston['Price'] = data.target

#Appreciation DIS 
if True:
    print(boston['DIS'].describe())
    #plt.hist(np.abs(stats.zscore(boston['DIS'])),100)
    plt.hist(boston['DIS'],200)
    plt.show()
    sns.boxplot(x=boston['DIS'])
    plt.show() 

#removing outliers and plotting 'em
if True:
    sd = np.std(boston['DIS'], axis=0)
    print(sd)
    DISprocessed =[x for x in boston['DIS'] if(x > np.mean(boston['DIS'],axis=0) - 2 * sd)]
    DISprocessed =[x for x in DISprocessed if(x < np.mean(boston['DIS'],axis=0) + 2 * sd)]
    sns.boxplot(x=DISprocessed)
    plt.show()
    plt.hist(DISprocessed,200)
    plt.show()

#Normmalizing and standardizing Boston data 
if True:
    normalizedBoston = preprocessing.normalize(boston)
    standardizedBoston = preprocessing.scale(boston)
    normPlusStandBoston = preprocessing.scale(normalizedBoston)


    
# Train and fit data 
if True:
    y = boston['Price']
    x = boston.drop('Price',axis=1)
    #sklearn.model_selection.train_test_split
    xtrain, xtest, ytrain,ytest= sklearn.model_selection.train_test_split(x,y,test_size=0.2)
    lm= LinearRegression()
    lm.fit(xtrain,ytrain)
    ypred = lm.predict(xtest) 
    ypred2 = lm.predict(xtrain)

 
 
# xtrain = Training set of x    
# xtest = Testing set of x
# ytrain = Training set of y  
# ytest = Testing set of y
# ypred = Predict set of y   
# ypred2 = Predict      
    
#Plot predictions ypred vis-a-vis ytest
t= np.arange(ypred.size)
plt.plot(t,ypred)
plt.plot(t,ytest)
plt.show()
plt.scatter(ypred,ytest) 
plt.show()
#plt.figure(figsize=(20,16))
plt.plot(t,ytest*0,'k')
plt.scatter(t,ytest-ypred) 
plt.show()

#Plot predictions ypred2 vis-a-vis ytrain
t= np.arange(ypred2.size)
plt.plot(t,ypred2)
plt.plot(t,ytrain)
plt.show()
plt.scatter(ypred2,ytrain) 
plt.show()
plt.plot(t,ytrain*0,'k')
plt.scatter(t,ytrain-ypred2) 
plt.show()

#Covariance , pearson and spearman
if True:
    cov = np.cov(boston['TAX'],boston['INDUS'])
    pearson, _ = pearsonr(boston['TAX'],boston['INDUS'])
    corr, _ = spearmanr(boston['TAX'],boston['INDUS'])
    plt.scatter(boston['TAX'],boston['INDUS'])
    plt.show()
    print(cov)
    print("Pearson")
    print(pearson)
    print("Spearman")
    print(corr)
    
    cov = np.cov(boston['NOX'],boston['DIS'])   
    pearson, _ = pearsonr(boston['NOX'],boston['DIS'])
    corr, _ = spearmanr(boston['NOX'],boston['DIS'])
    plt.scatter(boston['NOX'],boston['DIS'])
    plt.show()
    print(cov)
    print("Pearson")
    print(pearson)
    print("Spearman")
    print(corr)    
    plt.figure(figsize=(15,8))
    
    
    sns.heatmap(boston.corr(), annot=True)  
        
#Check Error values
if True:
    mse = sklearn.metrics.mean_squared_error(ytest, ypred)
    mae = sklearn.metrics.mean_absolute_error(ytest, ypred)
    print("MSE: %f, MAE: %f"%(mse,mae))





