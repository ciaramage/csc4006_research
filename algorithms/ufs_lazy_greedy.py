import math
import numpy as np
import array as arr
from helpers.algorithms.gram_schmidt import gram_schmidt

def lazy_greedy_UFS(X, Nc, rSquareMax=0.99):
    """ This function implements the Unsupervised Feature Selection algorithm
    with lazy greedy optimization applied.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        rSquareMax (float, optional): [description]. Defaults to 0.99.

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: The orthonormal basis used during each iteration after the first two components are selected
        rSquare: The smallest R squared value of each of the selected components
        compID: The component ID of each of the selected features 
    """
    #
    # Algorithm requires mean centeredcolumns
    #
    mX = X.mean(axis=1, keepdims=True)
    if max(mX) > 10**-6:
        # columns not mean centred
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    #
    # Setup
    #
    Y = X.copy()
    # reject columns with standard deviation < 0
    stdevs = np.std(Y, axis=0)
    stdevs_idx = np.where(stdevs < 0)
    Y = np.delete(Y, stdevs_idx, axis=1)
    stdevs = np.delete(stdevs, stdevs_idx)
    cols_idxs = np.arange(0, Y.shape[1]) # used to keep track of column features
    # normalize remaining columns to unit length
    Y = np.divide(Y, np.atleast_2d(stdevs))
    # setup storage vairables
    S = []
    M = []
    rSquare = []
    compID = arr.array('i')
    #
    # First component
    #
    corr_mat = np.corrcoef(np.matmul(Y.T, Y)) # correlation matrix
    sq_corr_mat = np.square(corr_mat) # squared correlation matrix
    # select as the first two columns those with the smallest squared correlation coefficient
    # and reject columns whose squared correlation coefficient with either exceeds rSquareMax
    smallest_sq_corr_mat = np.zeros(Y.shape[1])
    for i in range(0, Y.shape[1]):
        x = sq_corr_mat[:,i]
        idx = np.argmin(x)
        smallest_sq_corr_mat[i] =x[idx]
    # argsort(smallest_sq_corr_mat) returns indexes that put the smallest squared correlation
    # coefficient in ascending order
    idxs = np.argsort(smallest_sq_corr_mat)
    gains = smallest_sq_corr_mat[idxs]
    first_two_compIDs = idxs[0:2] 
    # append first two components to storage variables
    for ID in first_two_compIDs:
        S.append(Y[:,ID]) # each row in S is a column vector
    compID = np.append(compID, first_two_compIDs)
    rSquare = np.append(rSquare, gains[0:2])
    # find the pairwise correlation coefficient of each column with each of the 
    # selected columns and reject those > rSquareMax
    pc1, pc2 = pairwise_coefficient(compID, Y)
    idxs_reject = np.where((pc1 > rSquareMax) | (pc2 > rSquareMax))[0]
    Y = np.delete(Y, idxs_reject, axis=1)
    cols_idxs = np.delete(cols_idxs, idxs_reject)
    # remove the corresponding values of the rejected columns from gains and idxs
    gains_idxs_reject = [ i for i in range(len(idxs)) if idxs[i] in idxs_reject]
    idxs = np.delete(idxs, gains_idxs_reject)
    gains = np.delete(gains, gains_idxs_reject)
    #
    # Loop for remaining components
    #
    for j in range(0, Nc-2): # first two components have already been selected
        # Setup
        currentPos = j
        bestGain = gains[-1]
        bestGainIdx = 0
        worstGain = 0
        worstGainIdx = 0
        # find an orthonormal basis and project each remaining column of Y onto the subspace
        # spanned by the selected columns -> calculated using the gram_schmidt function
        # rows of S are the selected features -> transpose so selected features are represented by columns
        s = S.copy()
        Q = gram_schmidt(np.asarray(s).T)

        while True:
            # find the column index represented by the value at the current position of the idxs array
            colID = np.where(np.isin(cols_idxs, idxs[currentPos]))[0].item()
            x = Y[:, colID] 
            Rj = np.matmul(np.matmul(Q, Q.T), x)
            idx_min_Rj = np.argmin(Rj)
            gains[currentPos] = Rj[idx_min_Rj]

            # best gain
            if gains[currentPos] < bestGain:
                bestGain = gains[currentPos]
                bestGainIdx = idxs[currentPos]
            
            # worst gain
            if gains[currentPos] > worstGain:
                worstGain = gains[currentPos]
                worstGainIdx = idx[currentPos]
            
            if bestGain < gains[currentPos+1]:
                break # best gain found
            else:
                currentPos = currentPos + 1
                if currentPos == len(idxs) - 1:
                    break
        # append next component to data storage variables
        S.append(x)
        M.append(Q)
        compID = np.append(compID, bestGainIdx)
        rSquare = np.append(rSquare, bestGain)
        # resort list of gains and idxs
        if currentPos < len(idxs)-1:
            while worstGain > gains[currentPos+1]:
                currentPos = currentPos + 1
                if currentPos == len(idxs) - 1:
                    break
        newIdxs = np.argsort(gains[0:currentPos].flatten('C'))
        gains[0:currentPos] = idxs[newIdxs]
        idxs[0:currentPos] = idxs[newIdxs]
            
    return S, M, rSquare, compID

def pairwise_coefficient(cols, mat):
    """ This function calculates the pairwise correlation coefficient of the first two selected features
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

def do_ufs_lazy_greedy(X, Nc=1):
    """ This function returns the values calculated in the computation of the 
    UFS algorithm with lazy greedy optimization.

    Args:
        X (A 2D numpy array): Represents the dataset from which features are selected
        Nc (int, optional): The number of components to select. Defaults to 1.

    Returns:
       lazy_greedy_UFS(X, Nc): The results from performing
        Unsupervised Feature Selection with Lazy Greedy optimization on matrix X to select Nc columns
    """
    return lazy_greedy_UFS(X, Nc)   
