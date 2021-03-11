import math
import numpy as np 
from scipy.linalg import qr
from sklearn.metrics import r2_score
from helpers.algorithms.gram_schmidt import gram_schmidt

def ufs(X, Nc):
    # size of matrix X ( m - measurements, v - variables)
    m, v = X.shape

    if(Nc > v):
        Nc = v
    
    # matrix needs to have zero mean columns to be mean centred
    mX = X.mean(axis=1, keepdims=True)
    if(max(mX) > 10**-6):
        # column do not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    
    L = v # number of variables
    Y = X.copy()
    #
    # reject columns whose standard deviation is < 0
    #
    stdevs = np.std(Y, axis=1)
    std_idx = np.where(stdevs < 0)
    Y = np.delete(Y, std_idx, axis=1)
    stdevs = np.delete(stdevs, std_idx)
    #
    # normalise remaining columns to unit length
    #
    Y = np.divide(Y, stdevs)
    #
    # calculate correlation matrix
    #
    corr_mat = np.corrcoef(np.matmul(Y.T, Y))
    sq_corr_mat = np.square(corr_mat) # square correlation matrix
    #
    # smallest squared correlation coefficients in each column of Y
    # first two columns have the two smallest square correlation coefficient
    #
    smallest_sq_corr_mat = []
    for i in range(0, Y.shape[1]):
        x = sq_corr_mat[:,i] # column i
        idx = np.nanargmin(x)
        smallest_sq_corr_mat.append(x[idx]) 
    # value in k'th position is in it's sorted position, and values smaller than it go to the left
    k = np.argsort(np.argpartition(smallest_sq_corr_mat, axis=1)) # k -> new index order
    Y = Y[:,k] # update Y with new index order
    #
    # calculate coefficient of determination R^2 between selected columns and the remaining columns
    # find the value of maximum R^2 with each selected column
    #
    r2_col1 = np.zeros(Y.shape[1])
    r2_col2 = np.zeros(Y.shape[1])
    for i in range(2, Y.shape[1]):
        r2_col1[i] = r2_score(Y[:,0], Y[:,i])
        r2_col2[i] = r2_score(Y[:,0], Y[:,i])
    r2_col1_max = np.max(r2_col1)
    r2_col2_max = np.max(r2_col1)
    #
    # calculate the square pair wise correlation coefficient with selected columns
    # reject columns whose coefficient exceeds the R^2 maximum of either of the selected columns
    #
    pc_col1 = np.zeros(Y.shape[1])
    pc_col2 = np.zeros(Y.shape[1])
    for i in range(2, Y.shape[1]):
        pc_col1 = np.corrcoef(Y[:,0], Y[:,i])[0,1]
        pc_col2 = np.corrcoef(Y[:,1], Y[:,i])[0,1]
    idx_reject = np.where((pc_col1 > r2_col1_max) | (pc_col2 > r2_col2_max))
    Y = np.delete(Y, idx_reject, axis=1)

    #####
    # LOOP
    #####
    # the first two components/columns have already been selected
    for i in range( 2, Nc): 
        #
        # choose an orthonormal basis for the subspace R^P spanned by the first two columns
        # Q is a matrix whose columns are orthonormal
        #
        Q = gram_schmidt(Y, 0, 2)

        #
        # project each remaining column of Y onto the subspace spanned by the selected columns
        # which is calculated by the gram_schmidt function
        #
        Rj = np.zeros(Y.shape)
        for j in range(i, Y.shape[1]):
            rj = np.matmul(np.matmul(Q, Q.T), Y[:,j])
            Rj[:,i] = rj
        #
        # select the next column as that with the smallest Rj value
        # move the selected column into the appropriate position
        #
        smallest_Rj = []
        for j in range(0, Rj.shape[1]):
            x = Rj[:,j] # column j
            idx = np.nanargmin(x) # index of the smallest in each column
            smallest_Rj.append(x[idx])
        idx = np.nanargmin(smallest_Rj) # index of the next selected column
        Y[:,[i, idx]] = Y[:, [idx, 2]]

        








def do_ufs(X, Nc=1):
    return ufs(X, Nc)   
