import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1
#x_train = [number_of_samples,number_of_features] = number_of_samples x \in R^number_of_features

# Done programming and needs testing
def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)  # step 1
    index_lst=[]
    for it in xrange(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(xrange(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in xrange(dim):
            # gradient of loss function
            update_grad = ( logistic_wx(w[i],x) - y )*x*logistic_wx(w[i],x)*(1-logistic_wx(w[i],x))
            w[i] = w[i] - learn_rate*update_grad # update weights
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in xrange(niter):
        for i in xrange(dim):
            update_grad=0.0
            for n in xrange(num_n):
                x = x_train[n]
                y = y_train[n]
                update_grad += ( logistic_wx(w[i],x) - y )*x*logistic_wx(w[i],x)*(1-logistic_wx(w[i],x))
                update_grad+=(-logistic_wx(w,x_train[n])+y_train[n])# something needs to be done here
            w[i] = w[i] - learn_rate * update_grad/num_n
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=10):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.copper,edgecolors='black')

    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    error=[]
    y_est=[]
    for i in xrange(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',ax=ax,cmap=cm.coolwarm,edgecolors='black')
    print("error=",np.mean(error))
    return w

def get_data(problem):
    training_files = [ "data/data_big_nonsep_train.csv",
                       "data/data_big_separable_train.csv"
                       "data/data_small_nonsep_train.csv",
                       "data/data_small_separable_train.csv"]

    testing_files = ["data/data_big_nonsep_test.csv",
                     "data/data_big_separable_test.csv",
                     "data/data_small_nonsep_test.csv",
                     "data/data_small_separable_test.csv"]

    # Error handling
    if type(problem) != int:
        raise Exception("problem must be of type int")
    if problem < 0 or problem > 3 :
        raise Exception("problem must be in the interval (0,3)")

    # Construct training and testing sets
    xtrain = []
    ytrain = []
    with open(training_files[problem]) as f:
        testLine = f.readline()
        for line in f:
            xtrain.append(np.matrix(line.strip().split("\t")[:-1]))
            ytrain.append(np.matrix(line.strip().split("\t")[-1:]))

    xtest = []
    ytest = []
    with open(testing_files[problem]) as f:
        for line in f:
            xtest.append(np.matrix(line.strip().split("\t")[:-1]))
            ytest.append(np.matrix(line.strip().split("\t")[-1:]))


    return xtrain, ytrain, xtest, ytest

if __name__ == "__main__":
    data = get_data(0)

    test = np.hstack(np.array(data[0]))

    weights = train_and_plot(np.array(data[0]),
                             np.array(data[1]),
                             np.array(data[2]),
                             np.array(data[3]),
                             stochast_train_w)
