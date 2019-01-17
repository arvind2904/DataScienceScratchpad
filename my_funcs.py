#Typical functions required during playing with datasets for code brevity
#Auth: Prasad Ostwal
#Changelog
#   v1.0 -17Jan - Initial commit: Made functions from orignal temp.py
# Subsequent functions will be added to this file 

#Importing modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from datetime import date,datetime
import sklearn as sklearn
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import sklearn.metrics
from sklearn.datasets import load_digits
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy import stats
from scipy.stats import pearsonr,spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

#Function to calculate different error values 
def my_errors(d_orig,d_pred,title=None):
    mse = round(sklearn.metrics.mean_squared_error(d_orig, d_pred),2) 
    rmse = round(np.sqrt(sklearn.metrics.mean_squared_error(d_orig, d_pred)),2)
    mae = round(sklearn.metrics.mean_absolute_error(d_orig, d_pred),2)
    r2 = round(sklearn.metrics.r2_score(d_orig,d_pred),2)
    print("Error Values for %s:\n\t1.MSE: %s\n\t2.MAE: %s\n\t3.RMSE: %s\n\t4.R2: %s\n"%(title,mse,mae,rmse,r2))
    return mse,mae,rmse,r2

#Scatter function, so just calling one liner with args will plot the graph instead of multiple lines
# Need to call plt.show after calling this function. (So multiple data can be plotted and and then only once it will be showed )
#kwargs can be used to pass additional parameters
def my_scatter(x,y,title,xname,yname,size,color,**kwargs):
    plt.scatter(x,y,s=size,c=color,**kwargs)  
    plt.title(title)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()

# Pairgrid with time function. Will also return time required to plot 
def my_pairgrid(data,title,save=None):
    now = time.time()
    my_plot = sns.PairGrid(data)
    my_plot = my_plot.map_upper(plt.scatter,s=20)
    my_plot = my_plot.map_lower(sns.kdeplot, cmap="Blues_d")
    my_plot = my_plot.map_diag(plt.hist)     
    xlabels,ylabels = [],[]
    for ax in my_plot.axes[-1,:]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)        
    for ax in my_plot.axes[:,0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)        
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            my_plot.axes[j,i].xaxis.set_label_text(xlabels[i])
            my_plot.axes[j,i].yaxis.set_label_text(ylabels[j])    
    if save is True:
        my_plot.savefig("hunny %s"%(datetime.now().date().today()))
        print("Fig saved in root")
    return "Time required to plot graph is " + str(int(time.time()-now)) + " seconds."
    
#remove outliers from 'data' and plot boxplot if plot=True, 
def my_remove_outliers(data,plot,sdvalue):
         sd = np.std(data, axis=0)
         processed =[x for x in data if(x > np.mean(data,axis=0) - sdvalue * sd)]
         processed =[x for x in processed if(x < np.mean(data,axis=0) + sdvalue * sd)]
         if plot is True:
             sns.boxplot(data) #before
             plt.title("Before Processing")
             plt.show()
             sns.boxplot(processed) #after
             plt.title("After Processing")
             plt.show()

# This function is used for fetching variable name as string  during plotting axis names             
def get_name(variable):
 for name in globals():
     if eval(name) is variable:
         return name