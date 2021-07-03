import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing

def ufs(X, Nc):
    """ This function implements the baseline Unsupervised Feature Selection algorithm
    with no optimization applied.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
    Returns:
        S: The column vectors of each selected feature during each iteration
        M: The orthonormal basis used during each iteration after the first two components are selected
        rSquare: The smallest R squared value of each of the selected components
        compID: The component ID of each of the selected features 
    """
    # Normalise matrix columns to have zero mean and unit variance
    X = preprocessing.normalize(X, axis=0)
  
    # Correlation matrix X^T * X
    sq_corr = np.square(np.matmul(X.T, X))
    # Select as the first two columns those with the smallest squared correlation coefficient
    c_idxs = np.argpartition(np.min(sq_corr, axis=1), kth=1)[:2]

    # Keep track of column indexes not selected
    col_idxs = np.arange(X.shape[1])

    # Setup storage variables
    S = []
    M = []
    rSquare = []
    compID = []

    # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the first two columns
    # if the first two columns are Xa and Xb
    # c1 = Xa, and c2 = Y/|Y| - where Y = Xa - (Xb*Xa)*Xa
    c1 = np.atleast_2d(X[:,c_idxs[0]]).T
    Xb = np.atleast_2d(X[:,c_idxs[1]]).T
    c2 = c1 - np.dot(c1.T, Xb)*c1
    c2 = np.divide(c2, norm(c2))
    c = np.append(c1, c2, axis=1)

    compID = c_idxs
    rSquare.append(np.min(sq_corr, axis=1)[c_idxs])
    
    # update col_idxs by removing indexes of selected columns
    col_idxs = np.delete(col_idxs, c_idxs)

    # loop for remaining columns
    for _ in range(2, Nc):
        R = norm(np.matmul(np.matmul(c, c.T), X[:,col_idxs]), axis=0)
        idx = np.argmin(R)
        v = R[idx]
        compID = np.append(compID, col_idxs[idx])
        rSquare = np.append(rSquare, v)
        
        # For each remaining column, calculate its squared multiple correlation coefficient
        # R^2 wih the selected columns
        Xj = np.atleast_2d(X[:,col_idxs[idx]]).T
        ck = Xj - np.matmul(np.matmul(c, c.T), Xj)
        ck = np.divide(ck, norm(ck))
        # Update the orthonormal basis for the subspace spanned by the selected columns: c
        c = np.append(c, ck, axis=1)
        # Update col_idxs by removing the index of the column selected in the current iteration
        col_idxs = np.delete(col_idxs, idx)
    
    S = X[:,compID]
    M = c

    #return results
    return S, M, rSquare.tolist(), compID.tolist()
