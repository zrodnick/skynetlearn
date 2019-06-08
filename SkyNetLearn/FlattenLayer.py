import numpy as np

class flattenLayer:
    
    def __init__(self, indims, nodes):       
        pass
    
    def forward(self, A):
        self.dims = A.shape
        A = A.flatten().reshape(1,-1)
        return A
    
    def forwardDropout(self,A,p):
        self.dims = A.shape
        A = A.flatten().reshape(1,-1)
        return A
    
    def vanillaWU(self,LR,Dn,lmb1=0,lmb2=0):
        D = Dn.reshape(*self.dims)
        return D