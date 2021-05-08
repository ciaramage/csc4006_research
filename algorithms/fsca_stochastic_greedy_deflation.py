import numpy as np
from helpers.algorithms.gram_schmidt import gram_schmidt
from helpers.algorithms.fold_indices import split_with_replacement, split_without_replacement

def STOCHASTIC_FSCA_DEF( X, Nc, with_replacement, percentage=0.1):
    """ This function implements the Forward Selection Component Analysis algorithm with
    Stochastic Greedy (also known as lazier than lazy greedy) optimisation applied.
    At each iteration a random sample of the original data is taken and is used to 
    select the next feature. For more than one component, the indices of features selected
    in previous iterations are used to cover the features in the current subset by matrix deflation.
    

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        with_replacement (logical boolean, optiona;): Dictates whether random sampling
            should occur with or without replacement.
        percentage (int, optional): If random sampling occurs with replacement - this is the percentage
            of data selected from the original data. Defaults to 0.1

    Returns:
        S: The column vectors of each selected feature during each iteration
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features 
    """
    #
    # Setup
    #
    # matrix needs to be zero mean
    mX = X.mean(axis=1, keepdims=True)
    if(max(mX) > 10**-6):
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    Y = X.copy()
    #
    # split the data into Nc subsets
    #
    n_samples = Y.shape[0]
    n_splits = Nc
    subset_idxs =  split_with_replacement(n_samples, n_splits, 0.3) if with_replacement else split_without_replacement(n_samples, n_splits)
    #
    # initialise storage variables
    #
    S = []
    M = []
    VarEx = []
    compID = []
    TR = 0
    VEX = 0
    #
    # loop for components
    #
    rQ = np.zeros((Y.shape[1], 1))
    for j in range(0, Nc):
        # data subset for this iteration
        curr_idxs = subset_idxs[j] # current subset indices
        curr_Y = np.take(Y, curr_idxs, axis=0) # current subset columns
        TR = TR + np.trace(np.matmul(curr_Y.T, curr_Y))

        if j > 0: # if the component is not the first component
        # perform deflation step
            for ID in compID:
                x = np.atleast_2d(curr_Y[:,ID]).T
                th = np.matmul(np.linalg.pinv(x),curr_Y)
                Yhat = np.matmul(x, th)
                curr_Y = curr_Y-Yhat
        
        # maximise the rayleigh quotient
        for i in range(0, curr_Y.shape[1]):
            x = np.atleast_2d(curr_Y[:, i]).T
            r = np.matmul(curr_Y.T, x)
            rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
        idx = np.nanargmax(rQ)
        v = rQ[idx]
        vex = np.divide(100*v, TR)
        VEX = VEX + vex[0]

        # store results
        S.append(x)
        compID.append(idx)
        VarEx.append(VEX)

    return S, VarEx, compID