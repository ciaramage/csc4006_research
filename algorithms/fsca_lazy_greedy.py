import numpy as np 

from queue import PriorityQueue

# avoid reevaluating features with the least correlated rayLeigh quotient 
# apply minoux's lazy greedy optimization with a priority queue


def Lazy_Greedy_FSCA_analysis(X, Nc=1):
    #
    # size of matrix X (m - measurements, v - variables)
    #
    m = X.shape[0]
    v = x.shape[1]
    L = v # l is number of features (columns)
    #
    # if number of components < number of features in X 
    #   then number components = number of features in X
    #
    if Nc > m:
        Nc = m
    #
    # matrix x needs to have zero mean columns
    #
    mX = X.mean(axis=0) # calculate mean of X across axis=0 (0 for columns, 1 for rows)
    if( max(mX) > 10**-6):
        # columns do not have zero mean, detrend
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - np.ones(X.shape)*mX
    #
    # sum of matrix diagonal
    #
    Y = X
    TR = np.trace(np.matmul(Y.T, Y))
    #
    # iniitialise storage variables
    #
    compID = []
    VarEx = []
    S = []
    M = []
    VEX = 0
    #
    # initialize storage for rayleigh quotient
    #
    rQ = np.zeros((L,1))

    # first rayleigh quotient
    for i in range(0,L):
        x = Y[:,i] # column i
        x = np.atleast_2d(x).T
        # rayleigh quotient for x[i]
        r = np.matmul(Y.T, x)
        rQ[i] = (np.matmul(r.Y, np.divide(r, np.matmul(x.T, x))))
    #
    # convert rQ to variance explained
    #
    VEX = np.divide(100*v, TR)
    #
    # VEX -> order by descending, max > min
    #
    VEX = sorted(VEX, reverse=True)
    
