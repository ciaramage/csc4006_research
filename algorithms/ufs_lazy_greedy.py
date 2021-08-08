import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing


def ufs_lazy_greedy(X, Nc):
    """ This function implements the Unsupervised Feature Selection algorithm
    with lazy greedy optimization applied.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
    Returns:
        S: The column vectors of each selected feature during each iteration
        M: The orthonormal basis used during each iteration after the first two components are selected
        rSquare: The smallest R squared value of each of the selected components
        compID: The component ID of each of the selected features 
    """
    Y = X.copy()
    # Normalise matrix columns to have zero mean and unit variance
    Y = preprocessing.normalize(Y, axis=0) # axis=0 for column wise 

    # Correlation matrix Y^T * Y
    sq_corr = np.matmul(Y.T, Y)

    # Mask lower triangular so zero values aren't included in min function
    masked_upper = np.ma.masked_less_equal(np.triu(sq_corr), 0)
    
    # Column indexes of the two smallest squared correlation coefficient
    c_idxs = np.argpartition(np.min(masked_upper, axis=1), kth=1)[:2]

    # Keep track of column indexes not selected
    col_idxs = np.arange(Y.shape[1])

    # Setup storage variables
    M = []
    rSquare = []
    compID = []

    # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the first two columns
    # if the first two columns are Xa and Xb
    # c1 = Xa, and c2 = Y/|Y| - where Y = Xa - (Xb*Xa)*Xa

    compID = c_idxs
    rSquare.append(np.min(masked_upper, axis=1)[c_idxs]*100)
    
    # Update col_idxs by removing indexes of selected columns
    col_idxs = np.delete(col_idxs, c_idxs)

    # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the selected columns
    # if the first two columns are Xa and Xb, slide across each pair of columns
    # c1 = Xa, and c2 = Z/|Z| - where Z = Xa - (Xb*Xa)*Xa
    c = get_c(Y, compID)

    # The smallest square correlation coefficient of the remaining columns
    g = norm(np.matmul(np.matmul(c, c.T), Y[:,col_idxs]), axis=0)

    # argsort(g) returns the indices that put the correlation coefficient in ascending orer
    sorted_idx = np.argsort(g)

    # Put gains and corresponding column indexes in sorted order
    g = g[sorted_idx]
    gIdxs = col_idxs[sorted_idx]

    # Loop for remaining columns
    for i in range(0, Nc-2):
        pos = i # keep track of current position in list of gains: g
        bg = 1 # best gain
        bgIdx = 0 # best gain index
        wg = 0 # worst gain

        while True:
            # find the column represented by the current position in list of gains: g

            # check best gain: bg
            if g[pos] < bg:
                bg = g[pos]
                bgIdx = gIdxs[pos]
            
            # check worst gain: wg
            if g[pos] > wg:
                wg = g[pos]
            
            # evaluate best gain and current pos: bg, pos
            if bg < g[pos+1]:
                break # best gain found
            else:
                pos = pos+1
                if pos == len(gIdxs)-1:
                    break # position is at the last element
        
        # append to data storage
        compID = np.append(compID, bgIdx)
        rSquare = np.append(rSquare, bg*100)

        # resort the list of gains and indexes
        if pos < len(gIdxs)-1:
            while wg > g[pos+1]:
                pos = pos + 1
                if pos == len(gIdxs)-1:
                    break
        
        newIdxs = np.argsort(g[0:pos].flatten('C'))
        g[0:pos] = g[newIdxs]
        gIdxs[0:pos] = gIdxs[newIdxs]

       # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the selected columns
        # if the first two columns are Xa and Xb, slide across each pair of columns
        # c1 = Xa, and c2 = Z/|Z| - where Z = Xa - (Xb*Xa)*Xa
        c = get_c(Y, compID)

        # Update col_idxs by removing the index of the column selected in the current iteration
        #col_idxs = np.delete(col_idxs, idx)

    S = X[:,compID]
    M = c
    
    #return results
    return S, M, rSquare.tolist(), compID.tolist()

def get_c(X, idxs):
    c = np.atleast_2d(X[:,idxs[0]]).T
    for i in range(1, len(idxs)):
        Xi = np.atleast_2d(X[:,idxs[i]]).T
        ci = Xi - np.matmul(np.matmul(c, c.T), Xi)
        ci = np.divide(ci, norm(ci))
        c = np.append(c, ci, axis=1)
    return c
    
