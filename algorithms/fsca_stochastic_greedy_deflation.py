import numpy as np
from helpers.algorithms.gram_schmidt import gram_schmidt
from helpers.algorithms.fold_indices import split_with_replacement, split_without_replacement

def STOCHASTIC_FSCA_DEF( X, Nc, with_replacement, percentage=0):
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