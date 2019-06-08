import numpy as np

class sppLayer:
    
    def __init__(self, indims, nodes):
        pass
    
    def forward(self, A):
        self.dims=A.shape
        MP_quart=np.empty((4,A.shape[1]))
        maxlocsq = np.empty((4,A.shape[1]))
        step_q = int(A.shape[0]/4)
        qX = 0
        for i in range(0,int(A.shape[0]),step_q):
            MP_quart[qX,:] = np.max(A[i:i+step_q,:], axis=0)
            maxlocsq[qX,:] = np.argmax(A[i:i+step_q], axis=0)
            qX+=1
        MP_half=np.empty((2, A.shape[1]))
        maxlocsh = np.empty((2,A.shape[1]))
        step_h = int(A.shape[0]/2)
        hX = 0
        for i in range(0,int(A.shape[0]),step_h):
            MP_half[hX,:] = np.max(A[i:i+step_h,:], axis=0)
            maxlocsh[hX,:] = np.argmax(A[i:i+step_h,:], axis=0)
            hX+=1
        MP_all = np.max(A, axis=0)
        maxlocsf = np.argmax(A, axis=0)
        self.maxlocs = np.vstack((maxlocsq, maxlocsh, maxlocsf)).astype(int)
        An = np.vstack((MP_quart, MP_half, MP_all))
        return An
    
    def forwardDropout(self,A,p):
        self.dims=A.shape
        MP_quart=np.empty((4,A.shape[1]))
        maxlocsq = np.empty((4,A.shape[1]))
        step_q = int(A.shape[0]/4)
        qX = 0
        for i in range(0,int(A.shape[0]),step_q):
            MP_quart[qX,:] = np.max(A[i:i+step_q,:], axis=0)
            maxlocsq[qX,:] = np.argmax(A[i:i+step_q], axis=0)
            qX+=1
        MP_half=np.empty((2, A.shape[1]))
        maxlocsh = np.empty((2,A.shape[1]))
        step_h = int(A.shape[0]/2)
        hX = 0
        for i in range(0,int(A.shape[0]),step_h):
            MP_half[hX,:] = np.max(A[i:i+step_h,:], axis=0)
            maxlocsh[hX,:] = np.argmax(A[i:i+step_h,:], axis=0)
            hX+=1
        MP_all = np.max(A, axis=0)
        maxlocsf = np.argmax(A, axis=0)
        self.maxlocs = np.vstack((maxlocsq, maxlocsh, maxlocsf)).astype(int)
        An = np.vstack((MP_quart, MP_half, MP_all))
        return An
  

    def vanillaWU(self,LR,Dn,lmb1=0,lmb2=0):
        D = np.zeros(self.dims)
        S=int(self.dims[0]/4)
        cX=0
        for i in range(0,self.dims[0],S):
            for j in range(self.dims[1]):
                D[i:i+S,j][self.maxlocs[cX,j]] = Dn[cX,j]
            cX+=1
        return D        
        

       
        
        