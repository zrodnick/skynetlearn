#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def sumOfSquares(yhat, y):
    return np.sum((y-yhat).T@(y-yhat))/len(yhat)

def create_Phi(x):
    return np.hstack((np.ones((len(x),1)),x))


# In[ ]:


class LinearRegression:

    
    def __init__(self, X, Y, basis=create_Phi, closed=False):    
        self.basis = basis
        self.X = self.basis(X)
        if Y.shape[1] == 0:
            self.Y = np.array(Y).reshape(len(Y),1)
        else:
            self.Y = np.array(Y)
        self.target = self.Y.shape[1]
        self.features = self.X.shape[1]
        self.weights = np.random.randn(self.features, self.target)
        if closed == True: 
            self.w_best = np.linalg.solve(self.X.T.dot(self.X),self.X.T.dot(self.Y))
        
    def train(self, LR=[1e-6], epochs=1000, lmb1=[0], lmb2=[0], vocal=False, split=.8):
        data_index = np.array(range(0,len(self.X)))
        np.random.shuffle(data_index)
        sp_i = int(np.floor(len(self.X)*split))
        trn_i = data_index[0:sp_i]
        val_i = data_index[sp_i:]
        train = np.take(self.X, trn_i, axis=0)
        ytrn = np.take(self.Y, trn_i, axis=0)
        val = np.take(self.X, val_i, axis=0)
        yval = np.take(self.Y, val_i, axis=0)
        w = self.weights
        errors = []
        SS_best = np.inf
        for r in LR:
            for lm1 in lmb1:
                for lm2 in lmb2:
                    w = self.weights
                    for ep in range(epochs):
                        yhat = train@w
                        w -= r * train.T@(yhat-ytrn) - lm1*np.sign(w) - lm2*w
                        yvhat = val@w
                        SS = sumOfSquares(yvhat,yval)
                        errors.append([r, lmb1, lmb2, ep, SS])
                        if np.isnan(SS) | np.isinf(SS):
                            break
                        if vocal == True:
                            print("Lambda 1 / Lambda 2 / Epoch / Error = {}".format(errors[-1]))
                        if SS < SS_best:
                            SS_best = SS
                            self.parameters = [r, lmb1, lmb2]
                            self.w_best = w
        
                              
    def predict(self, test):
        self.predictions = self.basis(test)@self.w_best
        #self.prediction = (self.probabilities == self.probabilities.max(axis=1)[:,None]).astype(int)          

