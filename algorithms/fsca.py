import numpy as np
 
def fsca( X, Nc=1): 
    """ This function implements the baseline Forward Selection Component Analysis algorithm with 
    no optimization applied.
    
    Reference:
    L. Puggini, S. McLoone, Forward Selection Component Analysis: Algorithms
    and Applications, IEEE Transactions on Pattern Analysis and Machine Intelligence,
    Vol. 39(12), pp. 2395-2408, December 2017, DOI: 10.1109/TPAMI.2017.2648792.
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
    # Algorithm requires matrix X to have zero mean columns
    mX = X.mean(axis=0)
    if(max(mX) > 10**-6):
        # Columns not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX

    # Sum of matrix diagonal
    TR = np.trace( np.matmul(X.T, X))
    
    # Initialize storage variables
    S = []
    M = []
    compID = []
    VarEx = []
    VEX = 0
    
    # Keep track of columns not yet selected
    col_idxs = np.arange(X.shape[1])
    
    for _ in range(0, Nc):

        # Initialize storage for rayleigh quotient values 
        rQ= np.zeros(len(col_idxs))

        for i in range(len(col_idxs)):       
            # Column i
            x = np.atleast_2d(X[:,col_idxs[i]]).T 
            
            # Rayleigh quotient for column 
            r = np.matmul(X.T, x)
            rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
        # Maximise Rayleigh Quotient
        idx = np.nanargmax(rQ)
        v= rQ[idx]
        
        # Calculate accumulated variance explained
        VEX = VEX + np.divide(100*v, TR) # 100*v/TR = variance explained by selected component
        
        # Deflate matrix X
        x = np.atleast_2d(X[:,col_idxs[idx]]).T
        th = np.matmul(np.linalg.pinv(x),X)
        Yhat = np.matmul(x, th)
        X = X - Yhat
        
        # Store results
        M.append(th.T)
        compID.append(col_idxs[idx])
        VarEx.append(VEX)

        col_idxs = np.delete(col_idxs, idx)
    S = Y[:,compID]
    return S, M, VarEx, compID  