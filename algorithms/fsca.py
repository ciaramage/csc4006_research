import numpy as np
from sklearn import preprocessing
from numpy.linalg import eigh 
# Reference:
# L. Puggini, S. McLoone, Forward Selection Component Analysis: Algorithms
# and Applications, IEEE Transactions on Pattern Analysis and Machine Intelligence,
# Vol. 39(12), pp. 2395-2408, December 2017, DOI: 10.1109/TPAMI.2017.2648792.

# X is a matrix = m x v (measurements x variables)
# Nc is number of FSC components to compute


def FCSA_analysis( X, Nc=1): # Nc default to 1 if not defined in function call
    # matrix x needs to have zero mean columns - they should by default in matrix_generator but check anyway
    # TODO - Check zero mean columns for matrix
    mX = X.mean(axis = 0) # calculate mean of X across axis=0 (0 for columns, 1 for rows)
    if( max(mX) > 10**-6):
        # columns do not have zero mean then fix
        print("\nWarning: Data not zero mean... detrending\n")
        X = X- np.ones(X.shape)*mX

    #
    # size of matrix (m - measurements, v - variables)
    #

    m= X.shape[0] 
    v= X.shape[1]
    print(m)
    print(v)
    L = v # l is number of variables (columns)
    #
    # sum of matrix diagonal
    #
    Y = X
    Y_star = X.T
    TR = np.trace( np.matmul(Y, Y_star))
    #
    # initialize storage variables
    #
    compID = []
    VarEx = []
    YhatP = 0
    S = []
    M = []
    VEX = 0
    #
    # initialize storage for rayleigh quotient values 
    #
    rQ= np.zeros((L,1)) 

    for j in range(0,Nc):
        for i in range(0,L):
            #
            # x and x'
            #
            x = Y[:,i];# column i
            x_star = x.conjugate().T
     
            #
            # rayleigh quotient for x[i]
            #
            rQ[i] = (x_star.dot(Y_star.dot(x))) / (x_star.dot(x))
        #
        # select index of max rQ
        #
        print(rQ.shape)
        idx = np.argmax(rQ)
        #
        #accumulated variance explained
        #
        VEX = VEX + 100*v/TR  # 100*v/TR = variance explained by selected component

        #
        # deflate matrix
        #
        x = Y[:,idx]
        x = np.atleast_2d(x)
        print(x.shape)
        
        # theta
        th = np.matmul(Y, np.linalg.pinv(x))

        Yhat = x.dot(th)
        Y = Y-Yhat

        # store results
        S.append(x)
        M.append(th.T)
        YhatP = YhatP + Yhat
        compID.append(idx)
        VarEx.append(VEX)

    return S, M, VarEx, compID

def do_fsca(X):
    return FCSA_analysis(X,1)  

