import numpy as np

def norm(x):
    """L2 norm of column vector x

    Args:
        x (A numpy array): column vector stored in array

    Returns:
        A numpy array: L2 norm of the column vector x
    """
    return np.sqrt(np.sum(x * x))

def pca_first_nipals(X):
    """ Principal Component Analysis, implemented using NIPALS algorithm

    Args:
        X (2d numpy array): Matrix to find the first principal component of - using the NIPALS algorithm

    Returns:
        scores: The pca scores of first principal component calculated using NIPALS
    """
    tolerance = 1e-6 # tolerance for confergence
    v = X.shape[1] -1 # maximum column index
    idx = 0 # initial column index
    scores = []

    # select the first non-zero column of X as initial guess for scores
    x = np.atleast_2d(X[:,idx]).T
    xx = np.matmul(x.T, x)
    while xx < 5*np.finfo(float).eps and idx < v:
        idx = idx+1
        x = np.atleast_2d(X[:,idx]).T
        xx = np.matmul(x.T, x)
    
    if xx < 5*np.finfo(float).eps:
        print('***Warning... data has no non-zero columns') # all column variance  = 0
        return 
    
    scores = x 
    p = np.matmul(X.T, scores)
    p = np.divide(p, norm(p))
    tnew = np.matmul(X,p)

    # calculate largest component of x
    while norm(tnew-scores) > tolerance:
        scores = tnew
        p = np.matmul(X.T, scores)
        p = np.divide(p, norm(p))
        tnew = np.matmul(X,p)
    return scores
