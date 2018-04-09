import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from time import time

def logistic_z(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_wx(w,x):
    return logistic_z(np.inner(w,x))

def classify(w,x):
    x=np.hstack(([1],x))
    return 0 if (logistic_wx(w,x)<0.5) else 1

def stochast_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)  # step 1
    index_lst=[]
    for it in range(niter):
        if(len(index_lst)==0):
            index_lst=random.sample(range(num_n), k=num_n)
        xy_index = index_lst.pop()
        x=x_train[xy_index,:]
        y=y_train[xy_index]
        for i in range(dim):
            # gradient of loss function
            update_grad = ( logistic_wx(w,x) - y )*x[i]*logistic_wx(w,x)*(1-logistic_wx(w,x))
            w[i] = w[i] - learn_rate*update_grad # update weights
    return w

def batch_train_w(x_train,y_train,learn_rate=0.1,niter=1000):
    x_train=np.hstack((np.array([1]*x_train.shape[0]).reshape(x_train.shape[0],1),x_train))
    dim=x_train.shape[1]
    num_n=x_train.shape[0]
    w = np.random.rand(dim)
    index_lst=[]
    for it in range(niter):
        for i in range(dim):
            update_grad=0.0
            for n in range(num_n):
                x = x_train[n]
                y = y_train[n]
                update_grad += ( logistic_wx(w,x) - y )*x[i]*logistic_wx(w,x)*(1-logistic_wx(w,x))
            w[i] = w[i] - learn_rate * update_grad/num_n
    return w

def train_and_plot(xtrain,ytrain,xtest,ytest,training_method,learn_rate=0.1,niter=1000):
    plt.figure()
    #train data
    data = pd.DataFrame(np.hstack((xtrain,ytrain.reshape(xtrain.shape[0],1))),columns=['x','y','lab'])
    ax=data.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.cool,edgecolors='black')

    train_time = time()
    #train weights
    w=training_method(xtrain,ytrain,learn_rate,niter)
    train_time = time() - train_time
    print("training time:",train_time)
    error=[]
    y_est=[]
    for i in range(len(ytest)):
        error.append(np.abs(classify(w,xtest[i])-ytest[i]))
        y_est.append(classify(w,xtest[i]))
    y_est=np.array(y_est)
    data_test = pd.DataFrame(np.hstack((xtest,y_est.reshape(xtest.shape[0],1))),columns=['x','y','lab'])
    data_test.plot(kind='scatter',x='x',y='y',c='lab',cmap=cm.Wistia,edgecolors='black', ax=ax)

    print("error=",np.mean(error))
    plt.show()
    return w, np.mean(error), train_time

def get_data(problem):
    training_files = [ "data/data_big_nonsep_train.csv",
                       "data/data_big_separable_train.csv",
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
    xtrain = np.loadtxt(training_files[problem], delimiter="\t", usecols=(0,1))
    ytrain = np.loadtxt(training_files[problem], delimiter="\t", usecols=(2))

    xtest = np.loadtxt(testing_files[problem], delimiter="\t", usecols=(0,1))
    ytest = np.loadtxt(testing_files[problem], delimiter="\t", usecols=(2))

    return xtrain, ytrain, xtest, ytest

if __name__ == "__main__":
    problems = [1]  # Which problems to run
    iterations = [10,20,50,100,500]  # How many iterations will be used for training
    num_training_runs = 1  # Number of seperate training runs used to calculate avg error and training time

    # Run and log results
    errors, times = [[],[],[],[]], [[],[],[],[]]
    for prob in problems:
        print("Dataset:", prob)
        data = get_data(prob)

        for it, j in zip(iterations, range(len(iterations))):
            errors[prob].append([])
            times[prob].append([])
            for k in range(num_training_runs):
                print("Training run:", k)
                weights, it_error, it_time = train_and_plot(np.array(data[0]),
                                                            np.array(data[1]),
                                                            np.array(data[2]),
                                                            np.array(data[3]),
                                                            # batch_train_w,
                                                            stochast_train_w,
                                                            learn_rate=0.1,
                                                            niter=it)
                errors[prob][j].append(it_error)
                times[prob][j].append(it_time)

    # Print results
    print("----Avg----")
    for prob in problems:
        for it, i in zip(iterations, range(len(iterations))):
            print("Problem:", prob, "Iterations:", it)
            print("Avg. error:", np.mean(errors[prob][i]))
            print("Avg. times:", np.mean(times[prob][i]))
