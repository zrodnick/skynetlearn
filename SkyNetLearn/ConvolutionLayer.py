import numpy as np
from scipy.signal import correlate

def ReLU(x):
    return x*(x>0)

def dReLU(x):
    return x>0

class convolution:

    def __init__(self, indims, nodes):
        self.nodes = nodes
        self.indims = indims
        self.W = np.random.rand(5,self.indims,self.nodes)
        self.B = np.random.rand(1,self.nodes)  
    
    def forward(self, A):
        P = int((self.W.shape[0]-1)/2)
        H = np.empty((A.shape[0],self.nodes))
        self.A = np.vstack((np.zeros((P,A.shape[1])), A, np.zeros((P,A.shape[1]))))
        for i in range(self.nodes):
            H[:,i] = correlate(self.A, self.W[:,:,i], mode='valid')[:,0]
        H=H+self.B
        self.Z = ReLU(H)
        return self.Z
    
    def forwardDropout(self,A,p):
        P = int((self.W.shape[0]-1)/2)
        H = np.empty((A.shape[0],self.nodes))
        self.A = np.vstack((np.zeros((P,A.shape[1])), A, np.zeros((P,A.shape[1]))))
        for i in range(self.nodes):
            H[:,i] = correlate(self.A, self.W[:,:,i], mode='valid')[:,0]
        H=H+self.B
        self.Z = ReLU(H)
        return self.Z
    
    def vanillaWU(self,LR,Dn,lmb1=0,lmb2=0):
        P = int((self.W.shape[0]-1)/2)
        Dy = Dn*dReLU(self.Z)
        Dy = np.vstack((np.zeros((P,Dn.shape[1])), Dy, np.zeros((P,Dn.shape[1]))))
        Dx = np.empty((Dn.shape[0], self.indims))
        for k in range(self.indims):
            Dx[:,k] = correlate(Dy,self.W[::-1,k,:], mode='valid')[:,0]
            for l in range(self.nodes):
                self.W[:,k,l]-= LR*correlate(self.A[:,k], Dy[:,l], mode='valid')
                self.B-= LR*(np.sum(Dy, axis=0))
        return Dx
    
        