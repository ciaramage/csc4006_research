import numpy as np

def lazy_greedy_FSCA(X, Nc=1):
    """FSCA implemented with Minoux's lazy greedy optimization with a 
    priority queue. Avoids reevaluating features with the least correlated raleigh quotient.

    Args:
        X ([type]): [description]
        Nc (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    #
    # algorithm required to have zero mean columns
    #
    mX = X.mean(axis=1, keepdims=True)
    if(max(mX) > 10**-6):
        # column do not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    #
    # size of matrix (m - measurements, v - variables)
    #
    m,v = X.shape
    L = v # L is number of variables (columns)
    #
    # sum of matrix diagonal
    #
    Y = X
    TR = np.trace(np.matmul(Y.T, Y))
    #
    # first component
    #
    rQ = np.zeros((L,1))
    for i in range(0,L):
        x = np.atleast_2d(Y[:,i]).T # column i
        #
        # rayleigh quotient for first component
        #
        r = np.matmul(Y.T, x)
        rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
    VEX = 100 * np.divide(rQ, TR) # convert rayleigh quotient to variance explained
    #
    # argsort return indices the put variance in descending sorted order
    #
    idxs = np.argsort(VEX.flatten('C')) # indices to put the variance array in ascending order
    #
    # gains stored in priority queue in descending ordeer
    #
     
    
    



    #
    # initialize storage variables
    #
    S = []
    M = []
    VarEx = []
    compID = []


    return S, M, VarEx, compID