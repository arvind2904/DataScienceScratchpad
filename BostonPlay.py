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


# Make it false to save compiling time used to generate figures and if you dont want figures
Plot= False

# Load boston into Pandas Dataframe
data=load_boston()
boston = pd.DataFrame(data.data, columns=data.feature_names) 
boston['Price'] = data.target

#Appreciation DIS 
print(boston['DIS'].describe())
#plt.hist(np.abs(stats.zscore(boston['DIS'])),100)
plt.hist(boston['DIS'],200)
plt.show()
sns.boxplot(x=boston['DIS'])
plt.show() 

#removing outliers and plotting 'em
sd = np.std(boston['DIS'], axis=0)
print(sd)
DISprocessed =[x for x in boston['DIS'] if(x > np.mean(boston['DIS'],axis=0) - 2 * sd)]
DISprocessed =[x for x in DISprocessed if(x < np.mean(boston['DIS'],axis=0) + 2 * sd)]
sns.boxplot(x=DISprocessed)
plt.show()
plt.hist(DISprocessed,200)
plt.show()

#Normmalizing and standardizing Boston data 
normalizedBoston = preprocessing.normalize(boston)
standardizedBoston = preprocessing.scale(boston)
normPlusStandBoston = preprocessing.scale(normalizedBoston)

# Plot boston data Pairplot
if Plot is True:
    bostplot = sns.PairGrid(boston)
    #bostplot = bostplot.map(plt.scatter)
    bostplot = bostplot.map_upper(sns.scatter,s=20)
    bostplot = bostplot.map_lower(sns.kdeplot, cmap="Blues_d")
    bostplot = bostplot.map_diag(sns.hist) 
    
    xlabels,ylabels = [],[]
    
    for ax in bostplot.axes[-1,:]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)
        
    for ax in bostplot.axes[:,0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)
        
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            bostplot.axes[j,i].xaxis.set_label_text(xlabels[i])
            bostplot.axes[j,i].yaxis.set_label_text(ylabels[j])
    
    bostplot.savefig("hunny %s"%(datetime.now().date().today()))
    print("Flag: Plotting Complete")

# Remove outliers
    
    
# Train and fit data 
y = boston['Price']
x = boston.drop('Price',axis=1)
#sklearn.model_selection.train_test_split
xtrain, xtest, ytrain,ytest= sklearn.model_selection.train_test_split(x,y,test_size=0.2)
lm= LinearRegression()
lm.fit(xtrain,ytrain)
ypred = lm.predict(xtest) 


#Plot predictions
t= np.arange(ypred.size)
plt.plot(t,ypred)
plt.plot(t,ytest)
plt.show()
plt.scatter(ypred,ytest) 
plt.show()

#Check Error values
mse = sklearn.metrics.mean_squared_error(ytest, ypred)
mae = sklearn.metrics.mean_absolute_error(ytest, ypred)
print("MSE: %f, MAE: %f"%(mse,mae))





