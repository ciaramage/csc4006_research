import numpy as np

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

    # Size of sample
    sample_size = int(X.shape[0] * percentage)
    
    def get_sample_idxs():
        return np.random.randint(low = 0, high = X.shape[0], size = sample_size, dtype=int)
        
    # Initialise storage variables
    M = []
    varEx = []
    compID = []
    TR = 0
    vex = 0

    # Sample of original data for selecting first feature
    idxs = get_sample_idxs()
    Y = np.take(X, idxs, axis=0)

    TR = np.trace(np.matmul(Y.T, Y))

    # Initialise storage for rayleigh quotient values
    rQ = np.empty(Y.shape[1])

    # First component
    for i in range(Y.shape[1]):
        # Column i
        x = np.atleast_2d(Y[:,i]).T
        # Rayleigh quotient for column 
        r = np.matmul(Y.T, x)
        rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))

    # Maximise Rayleigh Quotient for first component
    idx = np.nanargmax(rQ)
    v = rQ[idx]

    # Calculate accumulated variance explained by selected component
    vex = vex + np.divide(100*v, TR)

    # Store first component results
    compID.append(idx)
    varEx.append(vex)

    # Loop for remaining components
    for i in range(1, Nc):
        # Update Rayleigh Quotient Storage
        rQ = np.empty(Y.shape[1])

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
        for j in range(Y.shape[1]):
            # Column i
            x = np.atleast_2d(Y[:,j]).T
            # Rayleigh quotient for column 
            r = np.matmul(Y.T, x)
            rQ[j] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))

        # Maximise Rayleigh Quotients
        idx = np.nanargmax(rQ)
        v = rQ[idx]

        # Calculate the accumulated variance explained with the inclusion of this feature
        vex = vex + np.divide(100*v, TR)

        # Store results
        compID.append(idx)
        varEx.append(vex)
        
    S = X[:,compID]

    return S, M, varEx, compID