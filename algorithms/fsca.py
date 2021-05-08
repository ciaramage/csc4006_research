import numpy as np
from numpy.linalg import eigh 
 
def FCSA( X, Nc=1): 
    """ This function implements the baseline Forward Selection Component Analysis algorithm with 
    no optimization applied.
    
    Reference:
    L. Puggini, S. McLoone, Forward Selection Component Analysis: Algorithms
    and Applications, IEEE Transactions on Pattern Analysis and Machine Intelligence,
    Vol. 39(12), pp. 2395-2408, December 2017, DOI: 10.1109/TPAMI.2017.2648792.
    Args:
        X (A 2D numppy array): The matrix m x v -> m is measurements, v is variables
        Nc (int, optional): The number of components to select. Defaults to 1.

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: Used to deflate the matrix at each iteration
        VarEx: The accumulated variance explained withb the inclusion of each selected feature
        compID: The component ID of each of the selected features 
    """
    # matrix needs to have zero mean columns to be mean centred
    mX = X.mean(axis=1, keepdims=True)
    if(max(mX) > 10**-6):
        # column do not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    #
    # size of matrix (m - measurements, v - variables)
    #
    m= X.shape[0] 
    v= X.shape[1]
    L = v # l is number of variables (columns)
    #
    # sum of matrix diagonal
    #
    Y = X
    TR = np.trace( np.matmul(Y.T, Y))
    #
    # initialize storage variables
    #
    compID = []
    VarEx = []
    S = []
    M = []
    VEX = 0
    #
    # initialize storage for rayleigh quotient values 
    #
    rQ= np.zeros((L,1)) 

    for j in range(0,Nc):
        for i in range(0,L):       
            x = np.atleast_2d(Y[:,i]).T # column i
            #
            # rayleigh quotient for x[i]
            #
            r = np.matmul(Y.T, x)
            rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
        #
        # select index of max rQ
        #
        idx = np.nanargmax(rQ)
        v= rQ[idx]
        #
        # accumulated variance explained
        #
        vex = VEX + np.divide(100*v, TR)
        VEX = vex[0]  # 100*v/TR = variance explained by selected component
        #
        # deflate matrix
        #
        x = Y[:,idx]
        x = np.atleast_2d(x).T
        th = np.matmul(np.linalg.pinv(x),Y)
        Yhat = np.matmul(x, th)
        Y = Y-Yhat
        #
        # store results
        #
        S.append(x)
        M.append(th.T)
        compID.append(idx) # component idx reflects matlab indexing from 1
        VarEx.append(VEX)
    return S, M, VarEx, compID  