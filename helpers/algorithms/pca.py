from statsmodels.multivariate.pca import PCA
import numpy as np

# return first components -> uses svd by default
# does not normalize, standardize, oe demean
def pca_first_components(X): # X = m x v, where m= no samples, v= no variables 
    pca = PCA(X, ncomp=1) 
    return pca.scores, len(pca.cols)

def pca_normalised(X, Nc=1):
    # X = m x v, where m= no samples, v= no variables
    pca = PCA(X, ncomp=Nc, normalize=True) # uses svd by default
    return pca.scores, len(pca.cols)

def norm(x):
    """L2 norm of column vector x

    Args:
        x ([array]): column vector stored in array

    Returns:
        [array]: L2 norm of column vector
    """
    return np.sqrt(np.sum(x * x))

def pca_first_nipals(X):
    """ Principal Component Analysis, implemented using NIPALS algorithm

    Args:
        X (2d matrix): 

    Returns:
        [array]: pca scores of first principak component calculated using NIPALS
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
