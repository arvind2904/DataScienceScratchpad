#Typical functions required during playing with datasets for code brevity
#Auth: Prasad Ostwal
#Changelog
#   v1.0 -17Jan - Initial commit: Made functions from orignal temp.py

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
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.datasets import load_digits
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr,spearmanr

#Function to calculate different error values 
def calc_errors(d_orig,d_pred,print_select):
    mse = sklearn.metrics.mean_squared_error(d_orig, d_pred)
    mae = sklearn.metrics.mean_absolute_error(d_orig, d_pred)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(d_orig, d_pred))
    r2 = sklearn.metrics.r2_score(d_orig,d_pred)
    if print_select is True:
        print("Error Values:\n\t1.MSE:%d\n\t2.MAE:%d\n\t3.RMSE:%d\n\t4.R2:%d\n"%(mse,mae,rmse,r2))
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
def my_pairgrid(data,save):
    now = time.time()
    my_plot = sns.PairGrid(data)
    my_plot = my_plot.map_upper(sns.scatter,s=20)
    my_plot = my_plot.map_lower(sns.kdeplot, cmap="Blues_d")
    my_plot = my_plot.map_diag(sns.hist)     
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
    exec_time = time.time() - now
    return exec_time
    