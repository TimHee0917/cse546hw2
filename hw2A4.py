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

def synData(n=500, d=1000, k =100, sig=1):
    eps = np.random.randn(n)
    x = np.random.randn(n, d)
    w = np.arange(1,d+1)/k
    w[ w > 1] = 0
    Y = (x).dot(w.T) + eps 	
    return(x, Y, w)


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

n=500;d=1000;k=100
X, Y, realW = synData(n=n, d=d, k=k)
lmax= lambdaMax(X, Y)
print("Lambda max is:{}".format(lmax))

Lambda=lmax
nonZero_list = []
FDR = []
TPR = []
Lambda_list = []
w = np.zeros(d)
while Lambda > 0.01:
    w,b = coord_desc_lasso(X, Y, Lambda, w=w, delta=0.01)
    nzero = (w>0).sum()
    nonZero_list.append( nzero )
    TPR.append( (w[0:k] > 0.0).sum()/k  )
    FDR.append( (w[k:] > 0.0).sum()/nzero  )
    Lambda_list.append(Lambda)
    Lambda = Lambda/1.5
        


#a
fig, ax = plt.subplots()
sns.lineplot(Lambda_list, nonZero_list)
sns.scatterplot(Lambda_list, nonZero_list)
plt.xscale('log')
ax.set_xlabel(" Labmda")
ax.set_ylabel("Number of non zeros")

#b
fig, ax = plt.subplots()
sns.lineplot(x=FDR, y=TPR, ax=ax)
sns.scatterplot(x=FDR, y=TPR, ax=ax)
ax.set_xlabel("FDR")
ax.set_ylabel("TPR")


