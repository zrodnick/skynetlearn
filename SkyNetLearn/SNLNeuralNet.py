#!/usr/bin/env python
# coding: utf-8


import numpy as np
from .RELU_Layer import *
from .Tanh_Layer import *
from .pRELU_Layer import *
from .Output_Layer import *
from .MaxPool import *
from .FlattenLayer import *
from .SpatialPyramidalPooling import *
from .ConvolutionLayer import *

actFuncs = {}
actFuncs['tanh'] = tanh
actFuncs['relu'] = RELU
actFuncs['prelu'] = pRELU
actFuncs['outlayer'] = outlayer
actFuncs['maxpool'] = MaxPool
actFuncs['flatten'] = flattenLayer
actFuncs['spp'] = sppLayer
actFuncs['convolution'] = convolution

def crossEntropy(p,y):
    if y.shape[1] <= 1:
        return -np.sum(np.multiply(y,np.log(p))+np.multiply((1-y),np.log(1-p)))/len(p)
    else:
        return -np.sum(np.multiply(y,np.log(p)))/len(p)

def sumOfSquares(yhat, y):
    if y.shape[1] <= 1:
        return (np.sum((y-yhat).T@(y-yhat)))/len(yhat)
    else:
        return (np.sum(np.trace((y-yhat).T.dot(y-yhat))))/len(yhat)

class NeuralNetwork:
    
    def __init__(self, indims, outdims, layers=['RELU','RELU','RELU'], nodes=[6,6,6],  
                 task='classification', cost=None):        
        self.indims=indims
        self.outdims=outdims
        self.task=task
        if cost:
            self.cost=cost
        elif task.lower() == 'classification':
            self.cost = crossEntropy
        elif task.lower() == 'regression':
            self.cost = sumOfSquares
        else:
            print('Invalid cost function')
            return None
        self.layers = layers
        self.layers.append('outlayer')
        self.lcount = len(layers)
        self.nodes = [indims]+nodes
        self.to_run = []
        for i in range(self.lcount-1):
            self.to_run.append(actFuncs[self.layers[i].lower()](indims=self.nodes[i], nodes=self.nodes[i+1]))
        self.to_run.append(actFuncs[self.layers[self.lcount-1].lower()](indims=self.nodes[self.lcount-1], nodes=self.outdims, task=self.task))
        
    def reinit(self):
        self.to_run = []
        for i in range(self.lcount-1):
            self.to_run.append(actFuncs[self.layers[i].lower()](indims=self.nodes[i], nodes=self.nodes[i+1]))
        self.to_run.append(actFuncs[self.layers[self.lcount-1].lower()](indims=self.nodes[self.lcount-1], nodes=self.outdims, task=self.task))        
        
    def split(self, X,Y, split=.8):
        data_index = np.array(range(0,len(X)))
        np.random.shuffle(data_index)
        sp_i = int(np.floor(len(X)*split))
        trn_i = data_index[0:sp_i]
        val_i = data_index[sp_i:]
        self.trn = np.take(X, trn_i, axis=0)
        self.ytrn = np.take(Y, trn_i, axis=0)
        self.val = np.take(X, val_i, axis=0)
        self.yval = np.take(Y, val_i, axis=0)        
    
    def everyday(self, X, Y):
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)
        
    def predict(self, X):
        A = X
        if self.NAG == True:
            for i in range(self.lcount):
                A = self.to_run[i].forwardNAG(A,self.mu)
        else:
            for i in range(self.lcount):
                A = self.to_run[i].forward(A)
        self.yhat = A
        if self.task == 'classification':
            if self.outdims == 1:
                self.predictions = np.rint(self.yhat)
            elif self.outdims > 1:
                self.predictions = (self.yhat == self.yhat.max(axis=1)[:,None]).astype(int)
        elif self.task == 'regression':
            self.predictions = self.yhat
            
    def trainDropout(self, X, p):
        A = X
        if self.NAG == True:
            for i in range(self.lcount):
                A = self.to_run[i].forwardDropoutNAG(A,p,self.mu)
        else:
            for i in range(self.lcount):
                A = self.to_run[i].forwardDropout(A,p)
        self.yhat = A
        
    def backprop(self, LR, y, lmb1, lmb2, ep, mu, gamma):
        D=y
        if self.method=='adagrad':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].adagrad(LR, D, lmb1, lmb2)
        elif self.method=='adagradm':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].adagradM(LR, D, lmb1, lmb2, mu)
        elif self.method=='adagradnag':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].adagradNAG(LR, D, lmb1, lmb2, mu)
        elif self.method=='rmsprop':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].rmsProp(LR, D, lmb1, lmb2, gamma)
        elif self.method=='rmspropm':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].rmsPropM(LR, D, lmb1, lmb2, mu, gamma)
        elif self.method=='rmspropnag':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].rmsPropNAG(LR, D, lmb1, lmb2, mu, gamma)
        elif self.method=='adam':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].adam(LR, D, lmb1, lmb2, ep, mu, gamma)
        elif self.method=='momentum':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].momentum(LR, D, lmb1, lmb2, mu)
        elif self.method=='momentumnag':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].momentumNAG(LR, D, lmb1, lmb2, mu)
        elif self.method=='basic':
            for i in range(self.lcount-1,-1,-1):
                D=self.to_run[i].vanillaWU(LR, D, lmb1, lmb2)
    
    
    def cnnPreProcess(self, X):
        xascii = [ord(i) for i in str(X)]
        x = np.zeros((len(xascii), 256))
        for i in range(len(xascii)):
            x[i,xascii[i]] = 1
        if len(x)%4==1:
            x = np.vstack((x, np.zeros((3,256))))
        elif len(x)%4==2:
            x = np.vstack((x, np.zeros((2,256))))
        elif len(x)%4==3:
            x = np.vstack((x, np.zeros((1,256))))
        self.xpr = x
        return self.xpr
        
    def cnntrain(self, X, Y, LR=1e-5, epochs=1000, vocal=False, lmb1=0, lmb2=0, dropout=None, mu=.9, gamma=.9, split=.8, method='basic', reinit=True, valstep=50):
        self.X = X
        self.Y = Y
        self.lmb1=lmb1
        self.lmb2=lmb2
        self.method = 'basic'
        self.mu=mu
        self.gamma=gamma
        if self.method.endswith('nag'):
            self.NAG = True
        else:
            self.NAG = False
        if split:
            self.split(self.X, self.Y, split=split)
        else:
            self.trn = self.X
            self.val = self.X
            self.ytrn = self.Y
            self.yval = self.Y

        self.everyday(self.trn, self.ytrn)
        self.trn_batches = np.array_split(self.trn, self.trn.shape[0])
        self.ytrn_batches = np.array_split(self.ytrn, self.ytrn.shape[0])
        
        self.everyday(self.val, self.yval)
        self.val_batches = np.array_split(self.val, self.val.shape[0])
        
        if reinit==True:
            self.reinit()        
        self.Errors = []
        runs = 1
        for ep in range(epochs):
            for xbatch, ybatch in zip(self.trn_batches, self.ytrn_batches):
                if dropout:
                    xtpr = self.cnnPreProcess(xbatch)
                    self.trainDropout(xtpr, dropout)
                else:
                    xtpr = self.cnnPreProcess(xbatch)
                    self.predict(xtpr)
                self.backprop(LR=LR, y=ybatch, lmb1=lmb1, lmb2=lmb2, ep=ep, mu=mu, gamma=gamma)
                if runs%valstep==0:
                    self.yhb = np.empty((0,self.outdims))
                    for vbatch in self.val_batches:
                        xvpr = self.cnnPreProcess(vbatch)
                        self.predict(xvpr)
                        self.yhb= np.vstack((self.yhb,self.yhat))
                    err = self.cost(self.yhb,self.yval)
                    self.Errors.append(err)
                    if vocal == True:
                        print('Iteration: {} / Error: {}'.format(runs, err))
                runs += 1
                
                
    def train(self, X, Y, LR=1e-5, epochs=1000, vocal=False, lmb1=0, lmb2=0, dropout=None, mu=.9, gamma=.9, split=.8, method='basic', batch='full', reinit=True):
        self.X = X
        self.Y = Y
        self.lmb1=lmb1
        self.lmb2=lmb2
        self.method = method.lower()
        self.mu=mu
        self.gamma=gamma
        if self.method.endswith('nag'):
            self.NAG = True
        else:
            self.NAG = False
        if split:
            self.split(self.X, self.Y, split=split)
        else:
            self.trn = self.X
            self.val = self.X
            self.ytrn = self.Y
            self.yval = self.Y
        if type(batch) is str:
            if batch == 'full':
                self.trn_batches = np.array_split(self.trn, 1)
                self.ytrn_batches = np.array_split(self.ytrn, 1)
            elif (batch == 'all') or (batch == 'stochastic'):
                self.everyday(self.trn, self.ytrn)
                self.trn_batches = np.array_split(self.trn, self.trn.shape[0])
                self.ytrn_batches = np.array_split(self.ytrn, self.ytrn.shape[0])
        elif type(batch) is int:
            if batch <= 1:
                self.everyday(self.trn, self.ytrn)
                self.trn_batches = np.array_split(self.trn, 1)
                self.ytrn_batches = np.array_split(self.ytrn, 1)
            else:
                self.everyday(self.trn, self.ytrn)
                self.trn_batches = np.array_split(self.trn, batch)
                self.ytrn_batches = np.array_split(self.ytrn, batch)
        if reinit==True:
            self.reinit()        
        self.Errors = []
        runs = 1
        for ep in range(epochs):
            for xbatch, ybatch in zip(self.trn_batches, self.ytrn_batches):
                if dropout:
                    self.trainDropout(xbatch, dropout)
                else:
                    self.predict(xbatch)
                self.backprop(LR=LR, y=ybatch, lmb1=lmb1, lmb2=lmb2, ep=ep, mu=mu, gamma=gamma)
                self.predict(self.val)
                err = self.cost(self.yhat,self.yval)
                self.Errors.append(err)
                if vocal == True:
                    print('Iteration: {} / Error: {}'.format(runs, err))
                runs += 1

        
  
                                   
