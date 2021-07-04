import numpy as np
from numpy.linalg import qr

def fsca_stochastic_greedy_deflation( X, Nc, percentage=0.5):
    """ This function implements the Forward Selection Component Analysis algorithm with
    Stochastic Greedy (also known as lazier than lazy greedy) optimisation applied.
    At each iteration a random sample of the original data is taken and is used to 
    select the next feature. For more than one component, the indices of features selected
    in previous iterations are used to cover the features in the current subset by matrix deflation.

    
    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        percentage (int, optional): If random sampling occurs with replacement - this is the percentage
            of data selected from the original data. Defaults to 0.5

    Returns:
        S: The column vectors of each selected feature during each iteration
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features 
    """
    # Algorithm requires matrix to have zero mean columns
    mX = X.mean(axis=0)
    if(max(mX) > 10**-6):
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    
    # Number of features (columns) in matrix X
    L = X.shape[1]

    # Size of sample
    sample_size = int(X.shape[0] * percentage)
    
    def get_sample_idxs():
        return np.random.randint(low = 0, high = X.shape[0], size = sample_size, dtype=int)
        
    # Initialise storage variables
    M = []
    VarEx = []
    compID = []
    TR = 0
    VEX = 0

    # Sample of original data for selecting first feature
    idxs = get_sample_idxs()
    Y = np.take(X, idxs, axis=0)

    TR = np.trace(np.matmul(Y.T, Y))

    # Keep track of columns not selected
    col_idxs = np.arange(X.shape[1])

    # Initialise storage for rayleigh quotient values
    rQ = np.zeros(len(col_idxs))

    # First component
    for i in range(len(col_idxs)):
        # Column i
        x = np.atleast_2d(Y[:,col_idxs[i]]).T
        # Rayleigh quotient for column 
        r = np.matmul(Y.T, x)
        rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))

    # Maximise Rayleigh Quotient for first component
    idx = np.nanargmax(rQ)
    v = rQ[idx]

    # Calculate accumulated variance explained by selected component
    VEX = VEX + np.divide(100*v, TR)

    # Store first component results
    compID.append(idx)
    VarEx.append(VEX)

    # Update col_idxs by removing the index of the selected column
    col_idxs = np.delete(col_idxs, idx)

    # Loop for remaining components
    for i in range(1, Nc):
        # Update Rayleigh Quotient Storage
        rQ = np.zeros(len(col_idxs))

        # Update sample
        idxs = get_sample_idxs()
        Y = np.take(X, idxs, axis=0)

        TR = np.trace(np.matmul(Y.T, Y))

        # Perform deflation step using the already selected columns
        for id in compID:
            x = np.atleast_2d(Y[:,id]).T
            th = np.matmul(np.linalg.pinv(x), Y)
            Yhat = np.matmul(x, th)
            Y = Y - Yhat
            M.append(th)

        # Calculate the Rayleigh Quotients
        # Initialise storage for rayleigh quotient values
        for j in range(len(col_idxs)):
            # Column i
            x = np.atleast_2d(Y[:,col_idxs[j]]).T
            # Rayleigh quotient for column 
            r = np.matmul(Y.T, x)
            rQ[j] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))

        # Maximise Rayleigh Quotients
        idx = np.nanargmax(rQ)
        v = rQ[idx]

        # Calculate the accumulated variance explained with the inclusion of this feature
        VEX = VEX + np.divide(100*v, TR)

        # Store results
        compID.append(col_idxs[idx])
        VarEx.append(VEX)

        # Update col_idxs by removing index of selected column
        col_idxs = np.delete(col_idxs, idx)
        
    S = X[:,compID]

    return S, M, VarEx, compID