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
    TR = np.trace( Y_star.dot(Y))
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
            x_star = np.atleast_2d(x)
            x = x_star.T
            # rayleigh quotient for x[i]
            #
            r = Y_star @ x
            r_star = r.conjugate().T
            rQ[i] = (r_star.dot(r)) / (x_star.dot(x)) 
 
        #
        # select index of max rQ
        #
        idx = np.argmax(rQ)
        #
        # variance explained
        v = Y[:,idx]
        # accumulated variance explained
        VEX += 100*v/TR  # 100*v/TR = variance explained by selected component

        #
        # deflate matrix
        #
        x = Y[:,idx]
        x = np.atleast_2d(x).T
        
        # theta
        th = np.linalg.pinv(x) @ Y

        Yhat = x.dot(th)
        Y = Y-Yhat

        # store results
        S.append(x)
        M.append(th.T)
        compID.append(idx)
        VarEx.append(VEX)

    return S, M, VarEx, compID

def do_fsca(X, Nc=1):
    return FCSA_analysis(X,Nc)  

