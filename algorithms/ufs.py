import numpy as np
from numpy.lib import corrcoef
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
    sq_corr = np.matmul(X.T, X)

    # Upper triangular of correlation matrix
    # Mask lower triangular so zero values aren't included in min function
    masked_upper = np.ma.masked_less_equal(np.triu(sq_corr), 0)
    
    # Select as the first two columns those with the smallest squared correlation coefficient
    c_idxs = np.argpartition(np.min(masked_upper, axis=1), kth=1)[:2]

    # Keep track of column indexes not selected
    col_idxs = np.arange(X.shape[1])

    # Setup storage variables
    M = []
    rSquare = []
    compID = []

    compID = c_idxs
    rSquare.append(np.min(masked_upper, axis=1)[c_idxs]*100)

    for _ in range(0, Nc - 2):
        # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the selected columns
        # if the first two columns are Xa and Xb, slide across each pair of columns
        # c1 = Xa, and c2 = Z/|Z| - where Z = Xa - (Xb*Xa)*Xa
        c = get_c(X, compID)

        # For each remaining column, calculate its squared multiple correlation coefficient
        # R^2 with the selected columns
        R = norm(np.matmul(np.matmul(c, c.T), X[:,col_idxs]), axis=0)
        idx = np.argmin(R)
        v = R[idx]
       
        compID = np.append(compID, col_idxs[idx])
        rSquare = np.append(rSquare, v*100)

        # Update col_idxs by removing the index of the column selected 
        # in the current iteration
        col_idxs = np.delete(col_idxs, idx)

        M = np.append(M, c)
    S = X[:,compID]
    return S, M, rSquare.tolist(), compID.tolist()

def get_c(X, idxs):
    c = np.atleast_2d(X[:,idxs[0]]).T
    for i in range(1, len(idxs)):
        Xi = np.atleast_2d(X[:,idxs[i]]).T
        ci = Xi - np.matmul(np.matmul(c, c.T), Xi)
        ci = np.divide(ci, norm(ci))
        c = np.append(c, ci, axis=1)
    return c