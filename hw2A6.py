import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mnist import MNIST
sns.set()
sns.set_style("ticks")
np.random.seed(0)
Lambda=0.1

def load_dataset():
    mndata = MNIST('./data/')
    mndata.gz=True
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0

    labels_train = labels_train.astype(np.int16)
    labels_test = labels_test.astype(np.int16)
    mask = ( (labels_test == 7) | (labels_test == 2) )
    labels_test = labels_test[mask]
    X_test = X_test[mask,:]
    labels_test[labels_test == 7]  = 1
    labels_test[labels_test == 2]  = -1

    mask = ( (labels_train == 7) | (labels_train == 2) )
    labels_train = labels_train[mask]
    X_train = X_train[mask,:]
    labels_train[labels_train == 7]  = 1
    labels_train[labels_train == 2]  = -1

    return X_train,labels_train, X_test, labels_test



	
def grad_desc(X, Y, w, b, eta):
    u = 1.0/(1.0+np.exp(-Y*(b + X.dot(w))))
    dtob=(-Y * (1-u)).mean()
    b = b -eta*dtob
    xy = np.multiply(X.T, Y)
    dtow = (- xy * (1-u)).mean(axis=1) + 2 * Lambda * w
    w = w-eta*dtow
    return(w,b)
	
def err(X, Y, w, b):
    inside = np.log( 1.0 + np.exp( -Y * (b + X.dot(w)) ) )
    j = inside.mean() + Lambda * np.linalg.norm(w,ord=2)
    pred = b + X.dot(w)
    pred[pred < 0] = -1
    pred[pred >= 0 ] = 1
    correct = np.sum(pred == Y)
    error = 1.0 - float(correct) / float(X.shape[0])
    return(j, error)

def makePlots(iters, test_j, train_j, test_e, train_e, name):
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(x=iters, y=test_j, ax=ax, label="Test")
	sns.lineplot(x=iters, y=train_j, ax=ax, label="Train")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("J(w,b)")
	plt.ylim(min(train_j)-.05, .5)
	plt.legend()
	
	fig, ax = plt.subplots(figsize=(16,9))
	sns.lineplot(x=iters, y=test_e, ax=ax, label="Test")
	sns.lineplot(x=iters, y=train_e, ax=ax, label="Train")
	ax.set_xlabel("Iteration")
	ax.set_ylabel("misclassification error")
	plt.ylim(0,.2)
	plt.legend()



def run(X_train, Y_train, X_test, Y_test, name, eta, itersize, batch=0):
    d=X_train.shape[1]
    w = np.zeros(d)
    b = 0
    iters = []; test_j=[]; train_j=[]; test_e = []; train_e = []
    j, error = err(X_train, Y_train, w, b)
    t_j, t_error = err(X_test, Y_test, w, b)
    test_j.append(t_j)
    train_j.append(j)
    test_e.append(t_error)
    train_e.append(error)
    iters.append(0)
    i = 1
    for k in range(0,itersize):
        n = X_train.shape[0]
        idx = np.random.permutation(n)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
        split = n/batch
        Xs = np.array_split(X_train, split)	
        Ys = np.array_split(Y_train, split)
        for X_split, Y_split in zip(Xs, Ys):
            w, b = grad_desc(X_split, Y_split, w, b, eta)
            j, error = err(X_train, Y_train, w, b)
            t_j, t_error = err(X_test, Y_test, w, b)
            test_j.append(t_j)
            train_j.append(j)
            test_e.append(t_error)
            train_e.append(error)
            iters.append(i)
            if(i % 100 == 0 or split == 1):
                print(j, error, i, k, X_split.shape)
            i += 1
    makePlots(iters, test_j, train_j, test_e, train_e, name)
		


X_train, Y_train, X_test, Y_test = load_dataset()
#run( X_train, Y_train, X_test, Y_test, "A6b", 0.01, 50, batch=X_train.shape[0]) 
run( X_train, Y_train, X_test, Y_test, "A6c", 0.001, 10, batch=1 ) 
#run( X_train, Y_train, X_test, Y_test, "A6d", 0.01, 10, batch=100 ) 




