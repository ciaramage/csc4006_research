import math
import array as arr
import numpy as np
from helpers.algorithms.gram_schmidt import gram_schmidt
from helpers.algorithms.fold_indices import split_with_replacement, split_without_replacement

def STOCHASTIC_UFS(X, Nc, rSquareMax = 0.99, with_replacement=True, percentage=0.1):
    #
    # Setup
    #
    Nc = 2 if Nc < 2 else Nc
    # algorithm requires mean centered columns
    m,v = X.shape
    mX = X.mean(axis=1, keepdims=True)
    if max(mX) > 10**-6:
        # columns not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    Y = X.copy()
    col_idxs = np.arange(0, Y.shape[1]) # used to keep track of column features

    # Split the data into Nc subsets
    n_samples = Y.shape[0]
    n_splits = Nc - 1 # minus 1 because the first split is used to find the first two components
    subset_idxs =  split_with_replacement(n_samples, n_splits, percentage) if with_replacement else split_without_replacement(n_samples, n_splits)

    # Initialise storage variables
    S = []
    rSquare = []
    compID = arr.array('i')

    #
    # First component
    #
    curr_idxs = subset_idxs[0] # current subset indices
    curr_Y = np.take(Y, curr_idxs, axis=0) # current subset data
    # Reject columns with standard deviation < 0
    stdev_idx = []
    stdevs = np.std(curr_Y, axis=0) # standard deviation along the columns
    stdevs_idx = np.where(stdevs < 0)
    curr_Y = np.delete(curr_Y, stdevs_idx, axis=1)
    stdevs = np.delete(stdevs, stdevs_idx)
    cols_idxs = np.arange(0, curr_Y.shape[1]) # used to keep track of column features
    curr_Y = np.divide(Y, np.atleast_2d(stdevs)) # normalize to unit length
    #
    # select as the first two columns those with the smallest squared correlation coefficient
    # and reject those whose squared correlation coefficient with either exceeds rSquareMax
    #
    corr_mat = np.corrcoef(np.matmul(curr_Y.T, curr_Y)) # correlation matrix
    square_corr_mat = np.square(corr_mat) # squared correlation matrix

    smallest_sq_corr_mat = [] # smallest squared correlation coefficient in each column
    for i in range(0, Y.shape[1]):
        x = square_corr_mat[:,i]
        idx = np.argmin(x)
        smallest_sq_corr_mat.append(x[idx])

    # the two smallest values in smallest_sq_corr_mat are used to select the first two components
    k = np.argpartition(smallest_sq_corr_mat, kth=1)[:2] # returns the indices of the 2 smallest square corrcoeff
    for kID in k:
        compID.append(int(kID))
        rSquare = np.append(rSquare, np.take(smallest_sq_corr_mat, k))

    for ID in compID:
        S.append(Y[:,ID]) # each row in S is a column vector for a selected column

    # find squared pairwise coefficient of each column with each selected column
    # and reject those > 
    pc1, pc2 = pairwise_coefficient(compID, curr_Y)
    idxs_reject = np.where((pc1 > rSquareMax) | (pc2 > rSquareMax))[0]
    curr_Y = np.delete(curr_Y, idxs_reject, axis=1)
    cols_idxs = np.delete(cols_idxs, idxs_reject)

    #
    # Loop for remaining columns
    #    
    idxs_to_mask = arr.array('i') # used to mask components already selected from argmin computation
    for i in range(1, n_splits): # from 1 as first split was used to select the first two components
        curr_idxs = subset_idxs[i] # update current subset indices
        curr_Y = np.take(Y, curr_idxs, axis=0) # current subset data
        selected_comps = np.take(curr_Y, compID, axis=1)
        curr_Y = np.delete(curr_Y, idxs_reject, axis=1)

        # Find an orthonormal basis and project each remaining column of curr_Y onto the subspace
        # spanned by the selected columns
        Q = gram_schmidt(selected_comps)
        Rj = np.zeros(curr_Y.shape)
        for j in range(0, curr_Y.shape[1]):
            Rj[:,j] = np.matmul(np.matmul(Q, Q.T),curr_Y[:,j])

        # Select as the next feature the column with the smallest Rj value, append its column ID to the data storage variable
        smallest_Rj = []
        for j in range(Rj.shape[1]):
            x = Rj[:,j] # column j
            idx = np.argmin(x) # index of the smallest Rj value in each column
            smallest_Rj.append(x[idx])

        # From the smallest Rj of the columns, which column has the smallest value
        smallest_Rj = np.asarray(smallest_Rj)
        smallest_Rj[idxs_to_mask] = np.nan
        idx = np.nanargmin(smallest_Rj) # index of the column with the smallest value in curr_Y
        next_ID = cols_idxs[idx] # original index of that column in X
        idxs_to_mask.append(idx)

        # Store results
        S.append(Y[:,idx])
        compID = np.append(compID, next_ID)
        rSquare = np.append(rSquare, smallest_Rj[idx])
    return S, rSquare, compID

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
    


   