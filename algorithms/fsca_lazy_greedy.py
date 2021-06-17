import numpy as np

def fsca_lazy_greedy(X, Nc=1):
    """ This function implements the Forward Selection Component Analysis algorithm with
    optimisation based on Minoux's lazy greedy optimization with a priority queue. 
    Avoids reevaluating features with the least correlated rayleigh quotient.
    This optimization uses arrays to implement a 'priority queue'

    Args:
        X (A 2D numppy array): The matrix m x v -> m is measurements, v is variables
        Nc (int, optional): The number of components to select. Defaults to 1.

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: Used to deflate the matrix at each iteration
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features
    """
    Y = X.copy()
    # Algorithm requires matrix to have zero mean columns
    mX = X.mean(axis=0)
    if(max(mX) > 10**-6):
        # Columns not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX

    # Number of features (columns) in matrix X
    L = X.shape[1] 
    
    # Sum of matrix diagonal
    TR = np.trace(np.matmul(X.T, X))
    
    # Initialise storage variables
    M = []
    VarEx = []
    compID = []
    VEX = 0

    # Initialise storage for rayleigh quotient values 
    rQ = np.zeros(L)

    for i in range(0,L):
        # Column i
        x = np.atleast_2d(X[:,i]).T 
        
        # Rayleigh quotient for first component
        r = np.matmul(X.T, x)
        rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))

    # Convert rayleigh quotient for first component to variance explained
    VEX = 100 * np.divide(rQ, TR) 
    
    # argsort(-array) returns the indices that put variances VEX in descending order
    idxs = np.argsort(-VEX.flatten('C'))
    g = VEX[idxs].flatten('C')
    firstCompId = idxs[0] # first index in idxs is the optimum one
    
    # Deflate matrix
    x = np.atleast_2d(X[:,firstCompId]).T
    th = np.matmul(np.linalg.pinv(x), X)
    YhatP = np.matmul(x, th)
    X = X - YhatP
    
    # Append first component data to storage variables
    M.append(th.T)
    VarEx.append(g[0])
    compID.append(firstCompId) 
    
    # Loop for the remaining components
    for j in range(1,Nc):
        pos = j # Keep track of the current position in list og gains: g
        bg = 0
        bgIdx = 0
        wg = g[0] 
        wgIdx = 0
        while True:
            x = np.atleast_2d(X[:, idxs[pos]]).T # column -> value at current position in idxs
            r = np.matmul(X.T, x)
            # increment in variance contribution with inclusion of this variable
            rQ = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
            g[pos] = 100 * np.divide(rQ, TR) # convert rayleigh quotient to variance explained

            # Check best gain: bg
            if g[pos] > bg:
                bg = g[pos]
                bgIdx = idxs[pos]

            # Check worst gain: wg
            if g[pos] < wg:
                wg = g[pos]
                wgIdx = idxs[pos]

            # If the best exact increment found to data is not greater than the upper bound on the 
            # next variable contribution, move to the next variable and evaluate it exactly
            if bg > g[pos+1]:
                break # best gain found
            else:
                pos = pos+1
                if pos == L-1:
                    break

        # Deflate matrix for selected feature
        x = np.atleast_2d(X[:, bgIdx]).T
        th = np.matmul(np.linalg.pinv(x), X)
        Yhat = np.matmul(x, th)
        X = X-Yhat
        
        # Store results for selected feature
        M.append(th.T)
        compID.append(bgIdx) 
        VarEx.append(bg + VarEx[-1])
        
        #  Resort the list of gains and indexes
        if pos < L-1: # last index position is L-1
            while wg < g[pos+1]:
                pos = pos+1
                if pos == L-1:
                    break
        newIdxs = np.argsort((-g[0:pos].flatten('C')))
        g[0:pos] = g[newIdxs]
        idxs[0:pos] = idxs[newIdxs]

    S = Y[:,compID]
    return S, M, VarEx, compID