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

    # Normalise matrix columns to have zero mean and unit variance
    X = preprocessing.normalize(X, axis=0) # axis=0 for column wise 

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
    
    # Update col_idxs by removing indexes of selected columns
    col_idxs = np.delete(col_idxs, c_idxs)

    # Lazy greedy part
    #################
    # the smallest square correlation coefficient of the remaining columns
    g = norm(np.matmul(np.matmul(c, c.T), X[:,col_idxs]), axis=0)
    # argsort(g) returns the indices that put the correlation coefficient in ascending orer
    sorted_idx = np.argsort(g)
    # put gains and corresponding columns indexes in sorted order
    g = g[sorted_idx]
    gIdxs = col_idxs[sorted_idx]

    # Loop for remaining columns
    for i in range(0, Nc-2):
        pos = i # keep track of current position in list of gains: g
        bg = g[-1] # best gain
        bgIdx = 0 # best gain index
        wg = 0 # worst gain
        wgIdx = 0 # worst gain index
        while True:
            # find the column represented by the current position in list of gains: g
            idx = np.where(np.isin(col_idxs, gIdxs[pos]))[0].item()
            R = norm(np.matmul(c, c.T) * X[:,idx])
            g[pos] = R

            # check best gain: bg
            if g[pos] < bg:
                bg = g[pos]
                bgIdx = gIdxs[pos]
            
            # check worst gain: wg
            if g[pos] > wg:
                wg = g[pos]
                wgIdx = gIdxs[pos]
            
            # evaluate best gain and current pos: bg, pos
            if bg < g[pos+1]:
                break # brest gain found
            else:
                pos = pos+1
                if pos == len(gIdxs)-1:
                    break # position is at the last element
        
        # append to data storage
        compID = np.append(compID, bgIdx)
        rSquare = np.append(rSquare, bg)

        # resort the list of gains and indexes
        if pos < len(gIdxs)-1:
            while wg > g[pos+1]:
                pos = pos + 1
                if pos == len(gIdxs)-1:
                    break
        newIdxs = np.argsort(g[0:pos].flatten('C'))
        g[0:pos] = g[newIdxs]
        gIdxs[0:pos] = gIdxs[newIdxs]

        # for each remaining column, calculate its square multiple correlation coefficient with the selected column
        Xj = np.atleast_2d(X[:,col_idxs[idx]]).T
        ck = np.subtract(Xj, np.matmul(np.matmul(c, c.T), Xj))
        ck = np.divide(ck, norm(ck))

        # update the orthonormal basis for the subspace spanned by the selected columns: c
        c = np.append(c, ck, axis=1)
    
    S = X[:,compID]
    M = c
    
    #return results
    return S, M, rSquare.tolist(), compID.tolist()


    
