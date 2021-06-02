import math
import array as arr
import numpy as np
from helpers.algorithms.gram_schmidt import gram_schmidt

def UFS(X, Nc, rSquareMax=0.99):
    """ This function implements the baseline Unsupervised Feature Selection algorithm
    with no optimization applied.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        rSquareMax (float, optional): The maximum coefficient of determination. Defaults to 0.99.

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: The orthonormal basis used during each iteration after the first two components are selected
        rSquare: The smallest R squared value of each of the selected components
        compID: The component ID of each of the selected features 
    """
    #
    # Algorithm requires mean centered columns
    #
    m,v = X.shape
    mX = X.mean(axis=1, keepdims=True)
    if max(mX) > 10**-6:
        # columns not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    #
    # Setup
    #
    Y = X.copy()

    # Reject columns with standard deviation < 0
    stdev_idx = []
    stdevs = np.std(Y, axis=0) # standard deviation along the columns
    stdevs_idx = np.where(stdevs < 0)
    Y = np.delete(Y, stdevs_idx, axis=1)
    stdevs = np.delete(stdevs, stdevs_idx)
    cols_idxs = np.arange(0, Y.shape[1]) # used to keep track of column features

    # Normalize remaining components to unit length
    Y = np.divide(Y, np.atleast_2d(stdevs))
    # Setup storage variables

    S = []
    M = []
    rSquare = []
    compID = arr.array('i')
    #
    # First component
    #
    corr_mat = np.corrcoef(np.matmul(Y.T, Y)) # correlation matrix
    square_corr_mat = np.square(corr_mat) # squared correlation matrix

    # select as the first two columns those with the smallest squared correlation coefficient
    # and reject columns whose squared correlation coefficient with either exceeds rSquareMax
    smallest_sq_corr_mat = [] # smallest squared correlation coefficient in each column
    for i in range(0, Y.shape[1]):
        x = square_corr_mat[:,i]
        idx = np.argmin(x)
        smallest_sq_corr_mat.append(x[idx])

    # the two smallest values in smallest_sq_corr_mat are used to select the first two components
    k = np.argpartition(smallest_sq_corr_mat, kth=1)[:2] # returns the indexes of the 2 smallest square corrcoef
    compID = np.append(compID, k)
    rSquare = np.append(rSquare, np.take(smallest_sq_corr_mat, k))
    for ID in compID:
        S.append(Y[:,ID]) # each row in S is a column vector

    # find squared pairwise correlation coefficient of each column with each selected column
    # and reject those > rSquareMax
    pc1, pc2 = pairwise_coefficient(compID, Y)
    idxs_reject = np.where((pc1 > rSquareMax) | (pc2 > rSquareMax))[0]
    Y = np.delete(Y, idxs_reject, axis=1)
    cols_idxs = np.delete(cols_idxs, idxs_reject)

    #
    # Loop for remaining components
    #
    idxs_to_mask = arr.array('i') # used to mask components already selected from argmin computation

    for i in range(2, Nc): # from 2 as the first two components have already been selected
        # Find an orthonormal basis and project each remaining column of Y onto the subspace
        # spanned by the selected columns -> calculated by gram_schmidt function
        # Rows of S are selected features -> transpose so selected features are represented by the columns
        Q = gram_schmidt(np.asarray(S).T)
        Rj = np.zeros(Y.shape)
        for j in range(0, Y.shape[1]):
            rj = np.matmul(np.matmul(Q, Q.T), Y[:,j])
            Rj[:,j] = rj
        
        # Select as the next feature the column with the smallest Rj value, append its column ID to the data storage variables
        smallest_Rj = []
        for j in range(0, Rj.shape[1]):
            x = Rj[:,j] # column j
            idx = np.argmin(x) # index of the smallest Rj value in each column
            smallest_Rj.append(x[idx])
        
        # From the smallest Rj of the columns, which column has the smallest value
        smallest_Rj = np.asarray(smallest_Rj)
        smallest_Rj[idxs_to_mask] = np.nan
        idx = np.nanargmin(smallest_Rj) # index of the column with the smallest value in Y
        next_ID = cols_idxs[idx] # original idx of that column in X -> this is the component ID
        idxs_to_mask.append(idx)

        # Store results
        S.append(Y[:,idx])
        M.append(Q)
        compID = np.append(compID, next_ID)
        rSquare = np.append(rSquare, smallest_Rj[idx])
    return S, M, rSquare, compID

def pairwise_coefficient(cols, mat):
    """This function calculates the pairwise correlation coefficient of the first two selected features
    represented by cols with each of the remaining features represented by mat

    Args:
        cols (A 2D numpy array): Represents the first two selected features.
        mat (A 2D numpy array): Represents the remaining features.

    Returns:
        pc_col1, pc_col2: The pairwise coefficient of the first (pc_col1) and the second (pc_col2) selected 
        features with each of the remaining features.
    """
    pc_col1 = np.zeros(mat.shape[1])
    pc_col2 = np.zeros(mat.shape[1])
    for i in range(0, mat.shape[1]):
        pc_col1[i] = np.corrcoef(mat[:, cols[0]],mat[:,i])[0][1]
        pc_col2[i] = np.corrcoef(mat[:, cols[1]], mat[:,i])[0][1]
    return pc_col1, pc_col2  