import math
import numpy as np 
import array as arr
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
    # initialise storage variables
    #
    compID = arr.array('i') # specifiy an array of integers
    S = []
    M = []
    smallestSquare = []

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
    Y = np.divide(Y, np.atleast_2d(stdevs).T)
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
    k = np.argpartition(smallest_sq_corr_mat, kth=1)[:2] # k contains the indexes of the 2 smallest square correlation coefficient
    compID = np.append(compID, k)
    smallestSquare = np.append(smallestSquare, np.take(smallest_sq_corr_mat, k))
    S.append(np.take(Y, k))
    #
    # calculate coefficient of determination R^2 between selected columns and the remaining columns
    # find the value of maximum R^2 with each selected column
    #
    r2_col1_max, r2_col2_max = coeffecient_determination(compID, Y)
    #
    # calculate the square pair wise correlation coefficient with selected columns
    # reject columns whose coefficient exceeds the R^2 maximum of either of the selected columns
    #
    pc_col1, pc_col2 = pairwise_coefficient(compID, Y)
    idx_reject = np.where((pc_col1 > r2_col1_max) | (pc_col2 > r2_col2_max))
    Y = np.delete(Y, idx_reject, axis=1)
    #####
    # LOOP
    #####

    # the first two components/columns have already been selected
    for i in range( 2, Nc): 
        #
        # choose an orthonormal basis for the subspace R^P spanned by the selected columns - indexes of which stored in compID
        # Q is a matrix whose columns are orthonormal
        #
        Q = gram_schmidt(Y, compID)
        #
        # project each remaining column of Y onto the subspace spanned by the selected columns
        # which is calculated by the gram_schmidt function
        #
        Rj = np.zeros(Y.shape)
        for j in range(0, Y.shape[1]):
            rj = np.matmul(np.matmul(Q, Q.T), Y[:,j])
            Rj[:,j] = rj
        #
        # select the next column as that with the smallest Rj value
        # append index to compID, and the column data to smallestSquare
        #
        smallest_Rj = []
        for j in range(0, Rj.shape[1]):
            x = Rj[:,j] # column j
            idx = np.nanargmin(x) # index of the smallest in each column
            smallest_Rj.append(x[idx])
        # dont want to include the index of columns already selected
        # set the value of index for columns already selected equal to nan
        for j in compID:
          smallest_Rj[j] = np.nan 
        ###
        idx = np.nanargmin(smallest_Rj) # index of the next selected column
        compID = np.append(compID, idx)
        smallestSquare = np.append(smallestSquare, smallest_Rj[idx])
        M.append(Q)
        S.append(Y[:,idx])
    return S, M, smallestSquare, compID




        
def coeffecient_determination(cols, mat ):
  r2_col1 = np.zeros(mat.shape[1])
  r2_col2 = np.zeros(mat.shape[1])
  for i in range(0, mat.shape[1]):
      r2_col1[i] = r2_score(mat[:,cols[0]], mat[:,i])
      r2_col2[i] = r2_score(mat[:,cols[1]], mat[:,i])
  r2_col1[cols[0]] = np.nan
  r2_col2[cols[1]] = np.nan
  r2_col1_max = np.nanmax(r2_col1)
  r2_col2_max = np.nanmax(r2_col2)
  return r2_col1_max, r2_col2_max   

def pairwise_coefficient(cols, mat):
    pc_col1 = np.zeros(mat.shape[1])
    pc_col2 = np.zeros(mat.shape[1])

    for i in range(0, mat.shape[1]):
        pc_col1 = np.corrcoef(mat[:, cols[0]],mat[:,i])[0,1]
        pc_col2 = np.corrcoef(mat[:,1], mat[:,i])[0,1]
    return pc_col1, pc_col2


def do_ufs(X, Nc=1):
    return ufs(X, Nc)   
