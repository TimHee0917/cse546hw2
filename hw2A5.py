import numpy as np
import scipy
import scipy.linalg as la
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
import pickle
import os

sns.set()
sns.set_style("ticks")
np.random.seed(0)

def lambdaMax(X, Y):
	diff = Y - np.mean(Y)
	lmax = 2*np.max(abs(diff.dot(X)))
	return(lmax)	


def coord_desc_lasso(X, Y, Lambda, w=None, delta = 0.001):
	diff = delta + 1
	d = X.shape[1]
	if(w is None):
		w = np.zeros(d)
	else:
		w = w.copy()
	
	aks = 2*np.sum(np.square(X), axis=0)

	while(delta < diff):
		b = 0
		w_original = w.copy()

		for k in np.arange(d):
			ak = aks[k]
			xk = X[:,k]	
			jnotk = X.dot(w.T) - w[k]*xk
			ck = 2*xk.dot(Y - (b + jnotk) )
			
			if(ck < -Lambda):
				w[k] = (ck + Lambda)/ak
			elif(-Lambda <= ck <= Lambda):
				w[k] = 0
			else:
				w[k] = (ck - Lambda)/ak
		
		diff = np.max(np.abs(w - w_original))
		

	return(w, b)

def MSE(Y, X, w, b):
	pred = X.dot(w.T) +b
	se = ((Y - pred) ** 2).mean()
	return(se)

import pandas as pd
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
train=df_train.drop("ViolentCrimesPerPop", axis = 1)        
X_train = df_train.drop("ViolentCrimesPerPop", axis = 1).values
Y_train = df_train["ViolentCrimesPerPop"].values
X_test = df_test.drop("ViolentCrimesPerPop", axis = 1).values
Y_test = df_test["ViolentCrimesPerPop"].values


lmax_train=lambdaMax(X_train, Y_train)
Lambda_train=lmax_train
print("Lambda_train max is:{}".format(lmax_train))
nonZero_list_train = []
Lambda_list_train = []
d_train = X_train.shape[1]
w_train = np.zeros(d_train)
w_train_list=[]
se_train_list=[]
se_test_list=[]
while Lambda_train >= 0.01:
    w_train,b_train = coord_desc_lasso(X_train, Y_train, Lambda_train, w=w_train, delta=0.001)
    nzero_train = (w_train>0).sum()
    nonZero_list_train.append( nzero_train )
    Lambda_list_train.append(Lambda_train)
    w_train_list.append(w_train)
    se_train=MSE(Y_train, X_train, w_train, b_train)
    se_test=MSE(Y_test, X_test, w_train, b_train)
    se_train_list.append(se_train)
    se_test_list.append(se_test)
    Lambda_train = Lambda_train/2.0

#a
fig, ax = plt.subplots()
sns.lineplot(Lambda_list_train, nonZero_list_train)
sns.scatterplot(Lambda_list_train, nonZero_list_train)
plt.xscale('log')
ax.set_xlabel(" Labmda_train")
ax.set_ylabel("Number of non zeros")

#b
a=list(train.columns).index("agePct12t29")
b=list(train.columns).index("pctWSocSec")
c=list(train.columns).index("pctUrban")
d=list(train.columns).index("agePct65up")
e=list(train.columns).index("householdsize")
lsta=[item[a] for item in w_train_list]
lstb=[item[b] for item in w_train_list]
lstc=[item[c] for item in w_train_list]
lstd=[item[d] for item in w_train_list]
lste=[item[e] for item in w_train_list]

fig, ax = plt.subplots()
sns.lineplot(Lambda_list_train, lsta)
sns.scatterplot(Lambda_list_train, lsta)
plt.xscale('log')
sns.lineplot(Lambda_list_train, lstb)
sns.scatterplot(Lambda_list_train, lstb)
plt.xscale('log')
sns.lineplot(Lambda_list_train, lstc)
sns.scatterplot(Lambda_list_train, lstc)
plt.xscale('log')
sns.lineplot(Lambda_list_train, lstd)
sns.scatterplot(Lambda_list_train, lstd)
plt.xscale('log')
sns.lineplot(Lambda_list_train, lste)
sns.scatterplot(Lambda_list_train, lste)
plt.xscale('log')
plt.show()
plt.legend(["agePct12t29","pctWSocSec","pctUrban","agePct65up","householdsize"],loc="upper right")

#c
fig, ax = plt.subplots()
sns.lineplot(Lambda_list_train, se_train_list)
sns.scatterplot(Lambda_list_train, se_train_list)
plt.xscale('log')
sns.lineplot(Lambda_list_train, se_test_list)
sns.scatterplot(Lambda_list_train, se_test_list)
plt.xscale('log')
plt.show()
plt.legend(["train error","test error"],loc="upper right")

#d
w_4,b_4=coord_desc_lasso(X_train, Y_train, 30, w=w_train, delta=0.001)
largest_loc=list(w_4).index(np.max(w_4))
least_loc=list(w_4).index(np.min(w_4))
print(list(train.columns)[largest_loc])
print(list(train.columns)[least_loc])