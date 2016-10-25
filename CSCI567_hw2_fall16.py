import numpy as np
import matplotlib.pyplot as plt
import math
import random

def splitting():
    f = open('housing.data.txt','r')
    testf = open('test.txt','w')
    trainf = open('train.txt','w')
    for i in range(506):
        if i%7 == 0:
            testw = f.readline()
            testf.write(testw)
        else:
            trainw = f.readline()
            trainf.write(trainw)
    f.close()
    testf.close()
    trainf.close()

def plotHist():
    train = np.loadtxt('train.txt')
    plt.figure(figsize=(16,10))
    for i in range(14):
        plt.subplot(4,4,i+1)
        plt.grid(True)
        x= train[:,i]
        plt.hist(x,bins=10,label = 'feature')
        plt.ylabel('numbers')
        plt.xlabel('feature')
    plt.show()
    return


def mean(x):
    sum = 0
    for i in range(len(x)):
        sum = x[i]+sum
    return sum/float(len(x))

def stand(x):
    sum = 0
    for i in range(len(x)):
        sum = pow((x[i] - mean(x)),2)+sum
    return math.sqrt(sum/float(len(x)-1))

def correlation(x,y):
    mx = mean(x)
    my = mean(y)
    sum1 = 0
    sum2 =0
    sum3 =0
    for i in range(len(x)):
        xu = x[i] - mx
        yu = y[i] - my
        sum1 = xu*yu+sum1
        sum2 = pow(xu,2)+sum2
        sum3 = pow(yu,2)+sum3
    return sum1/math.sqrt(sum2*sum3)

def featureWithHigh(train):
    h = np.array(train)
    cor ={}
    y = h[:,-1]
    for i in range(13):
        x = h[:,i]
        cor[i] = (abs(correlation(x,y)))
    cor = sorted(cor.iteritems(), key=lambda d:d[1], reverse = True)
    print cor
    return

def Standarlizaiton(traindata,test):
    m = len(traindata[0,:-1])
    for i in range(m):
        u = mean(traindata[:,i])
        s = stand(traindata[:,i])
        for j in range(len(test)):
            p = float(test[j,i]-u)/float(s)
            test = test.astype(float)    #need to pay attention to the type of different object
            test[j,i]= p
    return test

def linearRegressor(train):
    x = train[:,:-1]
    y = train[:,-1]
    w = np.linalg.pinv(x).dot(y)
    return w


def ypred(x,w):
    y=[]
    for i in range(len(x)):
        y.append(x[i,:].dot(w))
    #print np.array(y).reshape(len(x),1)
    return np.array(y).reshape(len(x),1)

def MSE(test,w):
    ytrue = test[:,-1]
    ytrue = ytrue.reshape(len(ytrue),1)
    sum = ((ypred(test[:,:-1],w) - ytrue)*(ypred(test[:,:-1],w) - ytrue)).sum(axis = 0)
    return sum/float(len(ytrue))


def testRegression(data,test):
    h = np.array(data)
    ht = Standarlizaiton(h,h)
    test = np.array(test)
    test = Standarlizaiton(h,test)    #???????????????   Caution: Need to use train mean and sd
    b = np.ones(len(ht))
    ht = np.c_[b,ht]
    b = np.ones(len(test))
    test = np.c_[b,test]
    c = linearRegressor(ht).reshape(len(data[0,:]),1)
    return MSE(test,c)

def RidgeRegressor(train,lamda):
    x = train[:,:-1]
    y = train[:,-1]
    i = np.eye(len(x[1,:]))
    w = np.linalg.inv(np.transpose(x).dot(x)+lamda*i).dot(np.transpose(x)).dot(y)
    #print w
    return w

def testRidgeRegression(train,test,lamda):
    h = np.array(train)
    ht = Standarlizaiton(h,h)
    test = np.array(test)
    test = Standarlizaiton(h,test)
    b = np.ones(len(ht))
    ht = np.c_[b,ht]
    b = np.ones(len(test))
    test = np.c_[b,test]
    c = RidgeRegressor(ht,lamda).reshape(14,1)
    return MSE(test,c)
#test = [[3,3],[2,2],[1,1]]
#test = np.array(test)
#b = np.ones(len(test))
#test = np.c_[b,test]
#print RidgeRegressor(test,0.007)

def CV(train):
    h = np.array(train)
    r = random.sample(range(433),433)
    cv = h[r[0],:].reshape(1,14)
    k = {}
    train ={}
    for i in range(1,433):
        cv = np.insert(cv, 0, values=h[r[i],:], axis=0)
    k[1] = cv[0:43,:]
    train[1] = cv[43:,:]
    k[2] = cv[43:86,:]
    train[2] = cv[0:43,:]
    train[2] = np.insert(train[2], 0, values=cv[86:,:], axis=0)
    k[3] = cv[86:129,:]
    train[3] = cv[0:86,:]
    train[3] = np.insert(train[3], 0, values=cv[129:,:], axis=0)
    k[4] = cv[129:172,:]
    train[4] = cv[0:129,:]
    train[4] = np.insert(train[4], 0, values=cv[172:,:], axis=0)
    k[5] = cv[172:215,:]
    train[5] = cv[0:172,:]
    train[5] = np.insert(train[5], 0, values=cv[215:,:], axis=0)
    k[6] = cv[215:258,:]
    train[6] = cv[0:215,:]
    train[6] = np.insert(train[6], 0, values=cv[258:,:], axis=0)
    k[7] = cv[258:301,:]
    train[7] = cv[0:172,:]
    train[7] = np.insert(train[7], 0, values=cv[215:,:], axis=0)
    k[8] = cv[301:344,:]
    train[8] = cv[0:301,:]
    train[8] = np.insert(train[8], 0, values=cv[344:,:], axis=0)
    k[9] = cv[344:387,:]
    train[9] = cv[0:344,:]
    train[9] = np.insert(train[9], 0, values=cv[387:,:], axis=0)
    k[10] = cv[387:,:]
    train[10] = cv[:387,:]
    return k,train

def showCorrelatioin(train):
    cor = {}
    for i in range(13):
        cor[i] = correlation(train[:,i],train[:,-1])
    print cor
    return

def Ridge1(train,test):
    lamda = 0.01
    print 'Ridge Regression:'
    while lamda<=1:
        print 'lmabda = %f:' %lamda
        print 'The MSE for training set is %f:'% testRidgeRegression(train,train,lamda)
        print 'The MSE for test set is %f:'% testRidgeRegression(train,test,lamda)
        lamda = lamda*10
    return

def RidgeWithCV(train,test):
    #avglamda = 0
    #for b in range(1):
    n = float(len(train))
    lamda = 0.0001
    k,ntrain = CV(train)
    #dd = {}
    mmin = 1000
    while lamda <= 10:
        f = []
        sum = 0
        print 'lmabda = %f:' % lamda
        for i in range(1,11):
            f.append(testRidgeRegression(ntrain[i],k[i],lamda))
            sum = f[i-1]+sum
            #print f
        #dd[lamda] = sum/float(10)
        #dd = min(dd.iteritems(), key=lambda d:d[1])
        aveMse = sum/float(10)
        if aveMse < mmin:
            clamda = lamda
            mmin = aveMse
        lamda = lamda*2
        print 'The MSE for CV resultes: %f' %aveMse
    #avglamda = clamda + avglamda
    #bestLamda = avglamda / float(10)
    print 'The lambda is: %f , when getting the best MSE.'% (clamda)
    print 'The MSE for test set is %f:'% testRidgeRegression(train,test,clamda)
    return




def linear2(train,test):
    print 'Correlation:', featureWithHigh(train)
    newtrain = np.c_[train[:,12],train[:,5]]
    newtrain = np.c_[newtrain,train[:,10]]
    newtrain = np.c_[newtrain,train[:,2]]
    newtrain = np.c_[newtrain,train[:,-1]]
    newtest = np.c_[test[:,12],test[:,5]]
    newtest = np.c_[newtest,test[:,10]]
    newtest = np.c_[newtest,test[:,2]]
    newtest = np.c_[newtest,test[:,-1]]
    print 'The MSE for training set is %f:'% testRegression(newtrain,newtrain)[0]
    print 'The MSE for test set is %f:'% testRegression(newtrain,newtest)[0]
    return

def find3b(train,test):
    featureWithHigh(train)
    newtrain = np.c_[train[:,12],train[:,-1]]
    h = np.array(newtrain)
    ht = Standarlizaiton(h,h)
    b = np.ones(len(ht))
    ht = np.c_[b,ht]
    c = linearRegressor(ht).reshape(len(newtrain[0,:]),1)
    y = ypred(ht[:,:-1],c)
    ytrue = newtrain[:,-1]
    ytrue = ytrue.reshape(len(ytrue),1)
    re = ytrue - y
    f2 = np.c_[train[:,:-1],re]
    featureWithHigh(f2)

    newtrain = np.c_[train[:,5],newtrain]
    h = np.array(newtrain)
    ht = Standarlizaiton(h,h)
    b = np.ones(len(ht))
    ht = np.c_[b,ht]
    c = linearRegressor(ht).reshape(len(newtrain[0,:]),1)
    y = ypred(ht[:,:-1],c)
    ytrue = newtrain[:,-1]
    ytrue = ytrue.reshape(len(ytrue),1)
    re = ytrue - y
    f3 = np.c_[train[:,:-1],re]
    featureWithHigh(f3)

    newtrain = np.c_[train[:,10],newtrain]
    h = np.array(newtrain)
    ht = Standarlizaiton(h,h)
    b = np.ones(len(ht))
    ht = np.c_[b,ht]
    c = linearRegressor(ht).reshape(len(newtrain[0,:]),1)
    y = ypred(ht[:,:-1],c)
    ytrue = newtrain[:,-1]
    ytrue = ytrue.reshape(len(ytrue),1)
    re = ytrue - y
    f4 = np.c_[train[:,:-1],re]
    featureWithHigh(f4)

    newtrain2 = np.c_[train[:,12],train[:,5]]
    newtrain2 = np.c_[newtrain2,train[:,10]]
    newtrain2 = np.c_[newtrain2,train[:,3]]
    newtrain2 = np.c_[newtrain2,train[:,-1]]
    newtest = np.c_[test[:,12],test[:,5]]
    newtest = np.c_[newtest,test[:,10]]
    newtest = np.c_[newtest,test[:,3]]
    newtest = np.c_[newtest,test[:,-1]]
    print 'The MSE for training set is %f:'% testRegression(newtrain2,newtrain2)[0]
    print 'The MSE for test set is %f:'% testRegression(newtrain2,newtest)[0]
    return

def BruteForce(train,test):
    count = 0
    min = 30
    for i in range(13):
        for j in range(i+1,13):
            for k in range(j+1,13):
                for t in range(k+1,13):
                    count = count+1
                    newtrain = np.c_[train[:,i],train[:,j],train[:,k],train[:,t],train[:,-1]]
                    #newtrain = np.c_[train[:,9],train[:,10],train[:,11],train[:,12],train[:,-1]]
                    #newtest = np.c_[test[:,i],test[:,j],test[:,k],test[:,t],test[:,-1]]
                    e = testRegression(newtrain,newtrain)[0]
                    #print e
                    if e <= min:
                        f1,f2,f3,f4 = i,j,k,t
                        min = e
    print f1+1,f2+1,f3+1,f4+1
    newtrain = np.c_[train[:,f1],train[:,f2],train[:,f3],train[:,f4],train[:,-1]]
    newtest = np.c_[test[:,f1],test[:,f2],test[:,f3],test[:,f4],test[:,-1]]
    print 'The MSE for training set is %f:'% testRegression(newtrain,newtrain)[0]
    print 'The MSE for test set is %f:'% testRegression(newtrain,newtest)[0]
    return

def polynomial(train,test):
    newtrain = train[:,:-1]
    newtest = test[:,:-1]
    for i in range(13):
        for j in range(13):
            new = train[:,i]*train[:,j]
            newtrain = np.c_[newtrain,new]
            te = test[:,i]*test[:,j]
            newtest = np.c_[newtest,te]
    newtrain = np.c_[newtrain,train[:,-1]]
    newtest = np.c_[newtest,test[:,-1]]
    print 'The MSE for training set is %f:'% testRegression(newtrain,newtrain)[0]
    print 'The MSE for test set is %f:'% testRegression(newtrain,newtest)[0]
    return

def main():
    train = np.loadtxt('train.txt')
    test = np.loadtxt('test.txt')
    print '.................Correlation............................'
    showCorrelatioin(train)
    print '.................Linear Regression:.....................'
    print 'The MSE for training set is %f:'% testRegression(train,train)[0]   #show training MSE
    print 'The MSE for test set is %f:'% testRegression(train,test)[0]    #show test MES
    print '.................Ridge Regression:.....................'
    print 'lambda = 0.01'
    print 'The MSE for training set is %f:'% testRidgeRegression(train,train,0.01/float(len(train)))[0]   #show training MSE
    print 'The MSE for test set is %f:'% testRidgeRegression(train,test,0.01/float(len(train)))[0]    #show test MES
    print 'lambda = 0.1'
    print 'The MSE for training set is %f:'% testRidgeRegression(train,train,0.1/float(len(train)))[0]   #show training MSE
    print 'The MSE for test set is %f:'% testRidgeRegression(train,test,0.1/float(len(train)))[0]    #show test MES
    print 'lambda = 1.0'
    print 'The MSE for training set is %f:'% testRidgeRegression(train,train,1.0/float(len(train)))[0]   #show training MSE
    print 'The MSE for test set is %f:'% testRidgeRegression(train,test,1.0/float(len(train)))[0]    #show test MES
    print '..............Ridge Regression with CV:................'
    RidgeWithCV(train,test)    #show test MES
    print '.................feature selection.....................'
    print 'Selection with correlation:'
    print '(a)'
    linear2(train,test)
    print '(b)'
    find3b(train,test)
    print 'Selection with Brute-force:'
    BruteForce(train,test)
    print '.................Polynomial Feature Expansion.....................'
    polynomial(train,test)
    plotHist()      #draw histogram
    return

main()
