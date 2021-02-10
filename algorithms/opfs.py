import numpy as np 
from sklearn import preprocessing
from numpy.linalg import eigh
from helpers.algorithms.pca import pca_nipals

def opfs(X, Nc=1):
    # matrix x needs to have zero mean columns - they should by default in matrix_generator but check anyway
    mX = X.mean(axis = 0) # calculate mean of X across axis=0 (0 for columns, 1 for rows)
    if( max(mX) > 10**-6):
        # columns do not have zero mean then fix
        print("\nWarning: Data not zero mean... detrending\n")
        X = X- np.ones(X.shape)*mX 
    
    #
    # size of matrix (m - measurements, v - variables)
    #
    (m,v) = X.shape 
    L = v # l is number of variables (columns)
    Y = X
    VT=Y.var(axis=0) # Y is column vector containing variance for each column

    #
    # initialize storage variables
    #
    compID = []
    VarEx = []
    YhatP = 0
    S = []
    M = []
    #
    # initialise storage for correlation vector
    #
    EFS=np.zeros((L,1))  # eigen factors

    for j in range(0,Nc):
        
        # calculate scores of 1st pc for Y using nipals algorithm
        t1, num_iterations  = pca_nipals(Y)

        for i in range(0,L):
            x = Y[:,i] # column i
            x_star = x.T
            
            EFS[i] = (x_star.dot(t1)**2) / (x_star.dot(x) + np.finfo(float).eps)
        # 
        # select index of max efs
        #
        idx = np.argmax(EFS)
        #
        # deflate matrix
        #
        x = Y[:,idx]
        x = np.atleast_2d(x)
        x = x.T
        # theta
        th = np.linalg.pinv(x) @ Y
        Yhat = x.dot(th)
        Y = Y-Yhat

        # 
        # store results
        #
        S.append(x)
        M.append(th.T)
        YhatP = YhatP + Yhat
        compID.append(idx)
        print('VT')
        print(VT)
        VarEx.append(np.var(np.divide(YhatP, VT*100)))
    
    # return S, M, VarEx, compId
    return S, M, VarEx, compID
    
def do_opfs(X, Nc=1):
    return opfs(X, Nc)          
