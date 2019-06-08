import numpy as np

class MaxPool:
    
    def __init__(self, indims, nodes):
        pass
                
    def forward(self,A):
        n1 = int(A.shape[0]/2)
        n2 = int(A.shape[1])
        Z=np.empty((n1,n2))
        self.maxlocs = np.empty((n1,n2))
        cX = 0
        for i in range(0,int(A.shape[0]),2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.maxlocs[cX,:] = np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z  

    
    def forwardNAG(self,A, mu=.9):
        n1 = int(A.shape[0]/2)
        n2 = int(A.shape[1])
        Z=np.empty((n1,n2))
        self.maxlocs = np.empty((n1,n2))
        cX = 0
        for i in range(0,int(A.shape[0]),2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.maxlocs[cX,:] = np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z  
    
    def forwardDropout(self,A,p):
        n1 = int(A.shape[0]/2)
        n2 = int(A.shape[1])
        Z=np.empty((n1,n2))
        self.maxlocs = np.empty((n1,n2))
        cX = 0
        for i in range(0,int(A.shape[0]),2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.maxlocs[cX,:] = np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z  
    
    def forwardDropoutNAG(self,A,p,mu=.9):
        n1 = int(A.shape[0]/2)
        n2 = int(A.shape[1])
        Z=np.empty((n1,n2))
        self.maxlocs = np.empty((n1,n2))
        cX = 0
        for i in range(0,int(A.shape[0]),2):
            Z[cX,:] = np.max(A[i:i+2,:], axis=0)
            self.maxlocs[cX,:] = np.argmax(A[i:i+2,:], axis=0)
            cX+=1
        return Z  
    
    def vanillaWU(self,LR,Dn,lmb1=0,lmb2=0):
        n1=int(Dn.shape[0]*2)
        n2=int(Dn.shape[1])
        out = np.zeros((n1,n2))
        D = np.zeros((n1,n2))
        cX=0
        for i in range(0,n1,2):
            for j in range(n1):
                D[i:i+2,j][self.maxlocs[cX,j]] = Dn[cX,j]
            cX+=1
        return D
        