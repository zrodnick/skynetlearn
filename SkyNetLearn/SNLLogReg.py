#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def create_Phi(x):
    return np.hstack((np.ones((len(x),1)),x))        

def softmax(h):
    phat = np.exp(h)/((np.sum(np.exp(h),axis = 1)).reshape(len(h),1))
    return phat

def sigmoid(h):
    return (1/(1 + np.exp(-h)))

def crossEntropy(p,y):
    if y.shape[1] <= 1:
        return -np.sum(np.multiply(y,np.log(p))+np.multiply((1-y),np.log(1-p)))/len(p)
    else:
        return -np.sum(np.multiply(y,np.log(p)))/len(p)


# In[ ]:


class LogisticRegression:
    
    def __init__(self, X, Y, basis=create_Phi):
        if Y.shape[1] == 1:
            self.outputfunc = sigmoid
            self.target = 1
        elif Y.shape[1] == 0:
            Y = Y.reshape(len(Y),1)
            self.outputfunc = sigmoid
            self.target = 1
        elif len(np.unique(Y)) > 1:
            self.outputfunc = softmax
            self.target = Y.shape[1]
        else:
            print("Invalid Dimensions")
        self.basis = basis
        self.X = np.array(self.basis(X))
        self.Y = np.array(Y)
        self.features = self.X.shape[1]          
    
    def initializeWeights(self):
        self.weights = np.random.randn(self.features, self.target) 
    
    def train(self, LR=[1e-6], epochs=1000, lmb1=[0], lmb2=[0], vocal=False, split=.8, reini=True):
        data_index = np.array(range(0,len(self.X)))
        np.random.shuffle(data_index)
        sp_i = int(np.floor(len(self.X)*split))
        trn_i = data_index[0:sp_i]
        val_i = data_index[sp_i:]
        train = np.take(self.X, trn_i, axis=0)
        ytrn = np.take(self.Y, trn_i, axis=0)
        val = np.take(self.X, val_i, axis=0)
        yval = np.take(self.Y, val_i, axis=0)
        if reini == True:
            self.initializeWeights()
        w = self.weights
        errors = []
        CE_best = np.inf
        for r in LR:
            for lm1 in lmb1:
                for lm2 in lmb2:
                    if reini==True:
                        self.initializeWeights()
                        w = self.weights
                    for ep in range(epochs):
                        p = self.outputfunc(train@w)
                        w -= r * train.T@(p-ytrn) - lm1*np.sign(w) - lm2*w
                        phat = self.outputfunc(val@w)
                        CE = crossEntropy(phat,yval)
                        errors.append([r, lmb1, lmb2, ep, CE])
                        if np.isnan(CE):
                            break
                        if vocal == True:
                            print("Lambda 1 / Lambda 2 / Epoch / Error = {}".format(errors[-1]))
                        if CE < CE_best:
                            CE_best = CE
                            self.parameters = [r, lmb1, lmb2]
                            self.w_best = w
        
                              
    def predict(self, test):
        self.predictions = self.outputfunc(self.basis(test)@self.w_best)
        #self.prediction = (self.probabilities == self.probabilities.max(axis=1)[:,None]).astype(int)          

