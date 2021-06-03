import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing

def UFS_new(X, Nc):
    # Normalise matrix columns to have zero mean and unit variance
    X = preprocessing.normalize(X, axis=0) # axis=0 for column wise 

    # Correlation matrix X^T * X
    sq_corr = np.matmul(X.T, X)
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
    g = np.min(sq_corr, axis=1)[col_idxs] 
    gIdxs = np.argsort(g)
    g = g[gIdxs]

    # loop for remaining columns
    for i in range(2, Nc):
        # Setup
        pos = i # keep track of current position in list of gains
        bg = g[-1] # best gain
        bgIdx = 0 # best gain index
        wg = 0 # worst gain
        wgIdx = 0 # worst gain index

        while True:
            # find the column represented by the current position in gIdxs
            id = np.where(np.isin(col_idxs, gIdxs[pos]))[0].item()
            R = norm(np.matmul(np.matmul(c, c.T), X[:,id]))
            g[pos] = R

            # check best gain bg
            if g[pos] < bg:
                bg = g[pos]
                bgIdx = gIdxs[pos]
            
            # check worst gain wg
            if g[pos] > wg:
                wg = g[pos]
                wgIdx = gIdxs[pos]

            # evaluate best gain and current position
            if bg < g[pos+1]:
                break # best gain found
            else: 
                pos = pos + 1
                if pos == len(gIdxs) -1:
                    break # if position is th elast position

        # append to data storage
        compID = np.append(compID, col_idxs[min_idx])
        rSquare = np.append(rSquare, R[min_idx])

        # resort list of gains and indexes
        if pos < len(gIdxs)-1:
            while wg > g[pos+1]:
                pos = pos + 1
                if pos == len(gIdxs) -1:
                    break
        newIdxs = np.argsort(g[0:pos].flatten('C'))
        g[0:pos] = gIdxs[newIdxs]
        gIdxs[0:pos] = gIdxs[newIdxs]
        

        # For each remanining column, calculate its squared multiple correlation coefficient
        # with the selected columns
        Xj = np.atleast_2d(X[:, col_idxs[min_idx]])
        ck = Xj - np.matmul(np.matmul(c, c.T), Xj)
        ck = np.divide(ck, norm(ck))
        # Update the orthormal basis for the subspace spanned by the selected columns: c
        c = np.append(c, ck, axis=1)


        
    S = X[:,compID]
    M = c

    #return results
    return S, M, rSquare.tolist(), compID.tolist()

