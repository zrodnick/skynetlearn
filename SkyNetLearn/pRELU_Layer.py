#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

def func(x,p=1):
    return x*(x>0) + p*x*(x<=0)

def derivFunc(x, p=1):
    return (x>0) + (x<=0)*p

def derivFuncp(x):
    return x*(x<=0)


class pRELU:
    
    def __init__(self, indims, nodes=6):
    #outdims is equal to the number of nodes in current layer, indims is dependent on previous layer.
        #weights    
        self.W = np.random.randn(indims, nodes) * np.sqrt(2/(indims+nodes))
        self.Mw = np.zeros_like(self.W)
        self.Gw = np.ones_like(self.W)
        self.Vw = np.zeros_like(self.W)
        self.Movw = np.zeros_like(self.W)
        #biases
        self.B = np.random.randn(1, nodes) * np.sqrt(2/(indims+nodes))
        self.Mb = np.zeros_like(self.B)
        self.Gb = np.ones_like(self.B)
        self.Vb = np.zeros_like(self.B)
        self.Movb = np.zeros_like(self.B)
        #p!
        self.P = np.ones_like(self.B)
        self.Mp = np.zeros_like(self.P)
        self.Gp = np.ones_like(self.P)
        self.Vp = np.zeros_like(self.P)
        self.Movp = np.zeros_like(self.P)
        
#####propagation methods#####

    def forward(self, X):
        self.A = X
        self.H = self.A@self.W+self.B
        self.Z = func(self.H)
        return self.Z
        #store As, Hs, and Zs
        
    def forwardDropout(self,X,p):
        self.A = X
        M = np.random.rand(*self.A.shape) < p
        Ahat = (self.A * M) / p
        self.H = Ahat@self.W+self.B
        self.Z = func(self.H)
        return self.Z
    
    def forwardDropoutNAG(self, X, p, mu=.9):
        self.W += mu*self.Mw
        self.B += mu*self.Mb
        self.P += mu*self.Mp
        self.A = X
        M = np.random.rand(*self.A.shape) < p
        Ahat = (self.A * M) / p
        self.H = Ahat@self.W+self.B
        self.Z = func(self.H)
        return self.Z
            
    def backward(self,Dn,lmb1=0,lmb2=0):
        #set derivative
        self.Regw = lmb1*np.sign(self.W) + lmb2*self.W
        self.Regb = lmb1*np.sign(self.B) + lmb2*self.B
        self.Regp = lmb1*np.sign(self.P) + lmb2*self.P
        deriv = Dn*derivFunc(self.H)
        self.gradw = self.A.T@deriv
        self.gradb = np.sum(deriv,axis=0)
        self.gradp = np.sum(Dn*derivFuncp(self.H),axis=0)
        self.D = deriv@self.W.T        
    
    def forwardNAG(self, X, mu=.9):
        self.W += mu*self.Mw
        self.B += mu*self.Mb
        self.P += mu*self.Mp
        self.A = X
        self.H = self.A@self.W+self.B
        self.Z = func(self.H)
        return self.Z
    
    def backwardNAG(self,Dn,lmb1=0,lmb2=0,mu=.9):
        #set derivative
        self.W -= mu*self.Mw
        self.B -= mu*self.Mb
        self.P += mu*self.Mp
        self.Regw = lmb1*np.sign(self.W) + lmb2*self.W
        self.Regb = lmb1*np.sign(self.B) + lmb2*self.B
        self.Regp = lmb1*np.sign(self.P) + lmb2*self.P
        deriv = Dn*derivFunc(self.H)
        self.gradw = self.A.T@deriv
        self.gradb = np.sum(deriv,axis=0)
        self.gradp = np.sum(Dn*derivFuncp(self.H),axis=0)
        self.D = deriv@self.W.T
        #call momentum after this

#####optimization methods#####
       
    def adagrad(self,LR,Dn,lmb1,lmb2):
        self.backward(Dn,lmb1,lmb2)
        self.Gw = self.Gw + (self.gradw)**2
        self.Gb = self.Gb + (self.gradb)**2
        self.Gp = self.Gp + (self.gradp)**2
        LRw = LR / np.sqrt(self.Gw+1e-10)
        LRb = LR / np.sqrt(self.Gb+1e-10)
        LRp = LR / np.sqrt(self.Gp+1e-10)
        self.W -= LRw*(self.gradw + self.Regw)
        self.B -= LRb*(self.gradb + self.Regb)
        self.P -= LRp*(self.gradp + self.Regp)
        return self.D
        
    def adagradM(self,LR,Dn,lmb1,lmb2,mu=.9):
        self.backward(Dn,lmb1,lmb2)
        self.Gw = self.Gw + (self.gradw)**2
        self.Gb = self.Gb + (self.gradb)**2
        self.Gp = self.Gp + (self.gradp)**2
        LRw = LR / np.sqrt(self.Gw+1e-10)
        LRb = LR / np.sqrt(self.Gb+1e-10)
        LRp = LR / np.sqrt(self.Gp+1e-10)
        self.Mw = mu*self.Mw - LRw*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LRb*(self.gradb + self.Regb)
        self.Mp = mu*self.Mp - LRp*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
     
    def adagradNAG(self,LR,Dn,lmb1,lmb2,mu=.9):
        self.backwardNAG(Dn,lmb1,lmb2,mu)
        self.Gw = self.Gw + (self.gradw)**2
        self.Gb = self.Gb + (self.gradb)**2
        self.Gp = self.Gp + (self.gradp)**2
        LRw = LR / np.sqrt(self.Gw+1e-10)
        LRb = LR / np.sqrt(self.Gb+1e-10)
        LRp = LR / np.sqrt(self.Gp+1e-10)
        self.Mw = mu*self.Mw - LRw*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LRb*(self.gradb + self.Regb)
        self.Mp = mu*self.Mp - LRp*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
        
    def rmsProp(self,LR,Dn,lmb1,lmb2,gamma=.9):
        self.backward(Dn,lmb1,lmb2)
        self.Gw = gamma*self.Gw + (1-gamma)*(self.gradw)**2
        self.Gb = gamma*self.Gb + (1-gamma)*(self.gradb)**2
        self.Gp = gamma*self.Gp + (1-gamma)*(self.gradp)**2
        LRw = LR / np.sqrt(self.Gw+1e-10)
        LRb = LR / np.sqrt(self.Gb+1e-10)
        LRp = LR / np.sqrt(self.Gp+1e-10)
        self.W -= LRw*(self.gradw + self.Regw)
        self.B -= LRb*(self.gradb + self.Regb)
        self.P -= LRp*(self.gradp + self.Regp)
        return self.D
                 
    def rmsPropM(self,LR,Dn,lmb1,lmb2,mu=.9, gamma=.9):
        self.backward(Dn,lmb1,lmb2)
        self.Gw = gamma*self.Gw + (1-gamma)*(self.gradw)**2
        self.Gb = gamma*self.Gb + (1-gamma)*(self.gradb)**2
        self.Gp = gamma*self.Gp + (1-gamma)*(self.gradp)**2
        LRw = LR / np.sqrt(self.Gw+1e-10)
        LRb = LR / np.sqrt(self.Gb+1e-10)
        LRp = LR / np.sqrt(self.Gp+1e-10)
        self.Mw = mu*self.Mw - LRw*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LRb*(self.gradb + self.Regb)
        self.Mp = mu*self.Mp - LRp*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
        
    def rmsPropNAG(self,LR,Dn,lmb1,lmb2,mu=.9, gamma=.9):
        self.backwardNAG(Dn,lmb1,lmb2,mu)
        self.Gw = gamma*self.Gw + (1-gamma)*(self.gradw)**2
        self.Gb = gamma*self.Gb + (1-gamma)*(self.gradb)**2
        self.Gp = gamma*self.Gp + (1-gamma)*(self.gradp)**2
        LRw = LR / np.sqrt(self.Gw+1e-10)
        LRb = LR / np.sqrt(self.Gb+1e-10)
        LRp = LR / np.sqrt(self.Gp+1e-10)
        self.Mw = mu*self.Mw - LRw*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LRb*(self.gradb + self.Regb)
        self.Mp = mu*self.Mp - LRp*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
        
    def adam(self,LR,Dn,lmb1,lmb2,ep, mu=.9, gamma=.9):
        self.backward(Dn,lmb1,lmb2)
        self.Movw = mu*self.Movw + (1-mu)*(self.gradw)
        self.Movb = mu*self.Movb + (1-mu)*(self.gradb)
        self.Movp = mu*self.Movp + (1-mu)*(self.gradp)
        self.Vw = gamma*self.Vw + (1-gamma)*(self.gradw)**2
        self.Vb = gamma*self.Vb + (1-gamma)*(self.gradb)**2
        self.Vp = gamma*self.Vp + (1-gamma)*(self.gradp)**2
        movwhat = self.Movw / (1 + mu**(ep+1))
        movbhat = self.Movb / (1 + mu**(ep+1))
        movphat = self.Movp / (1 + mu**(ep+1))
        vwhat = self.Vw / (1 - gamma**(ep+1))
        vbhat = self.Vb / (1 - gamma**(ep+1))
        vphat = self.Vp / (1 - gamma**(ep+1))
        LRw = LR / np.sqrt(vwhat+1e-10)
        LRb = LR / np.sqrt(vbhat+1e-10)
        LRp = LR / np.sqrt(vphat+1e-10)
        self.W -= LRw*(movwhat + self.Regw)
        self.B -= LRb*(movbhat + self.Regb)
        self.P -= LRp*(movphat + self.Regp)
        return self.D
    
    def momentum(self,LR,Dn,lmb1,lmb2, mu): 
        self.backward(Dn,lmb1,lmb2)
        self.Mw = mu*self.Mw - LR*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LR*(self.gradb + self.Regb)
        self.Mp = mu*self.Mp - LR*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
        
    def momentumNAG(self,LR,Dn,lmb1,lmb2,mu): 
        self.backwardNAG(Dn,lmb1,lmb2)
        self.Mw = mu*self.Mw - LR*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LR*(self.gradb + self.Regb)
        self.Mp = mu*self.Mp - LR*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
        
    def momentumDecay(self,LR,Dn,lmb1,lmb2, mu):
        self.backward(Dn,lmb1,lmb2)
        self.Mw = mu*self.Mw - LR*(self.gradw + self.Regw)
        self.Mb = mu*self.Mb - LR*(self.gradb + self.Regb) 
        self.Mp = mu*self.Mp - LR*(self.gradp + self.Regp)
        self.W += self.Mw
        self.B += self.Mb
        self.P += self.Mp
        return self.D
        
    def vanillaWU(self,LR,Dn,lmb1,lmb2):
        self.backward(Dn,lmb1,lmb2)
        self.W -= LR*(self.gradw + self.Regw)
        self.B -= LR*(self.gradb + self.Regb)
        self.P -= LR*(self.gradp + self.Regp)
        return self.D
                






