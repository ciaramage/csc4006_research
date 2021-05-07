import numpy as np 
from sklearn import preprocessing
from helpers.algorithms.pca import pca_first_nipals
from helpers.algorithms.gram_schmidt import gram_schmidt
from helpers.algorithms.fold_indices import split_with_replacement, split_without_replacement

def STOCHASTIC_OPFS(X, Nc=1, with_replacement=True, percentage=0):

    #
    #  Setup
    #
    # matrix needs to be zero mean
    mX = X.mean(axis=1, keepdims=True)
    if(max(mX) > 10**-6):
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    Y = X.copy()
    # split the data into Nc subsets
    n_samples = Y.shape[0]
    n_splits = Nc
    subset_idxs =  split_with_replacement(n_samples, n_splits, 0.3) if with_replacement else split_without_replacement(n_samples, n_splits)
    #
    # Initialize storage variables
    #
    S = []
    VarEx = []
    compID = []
    VEX = 0
    VT = 0
    YhatP = 0
    EFS = np.zeros((Y.shape[1], 1))
    #
    # Loop for components
    #
    for j in range(0, Nc):
        # data subset for this iteration
        curr_idxs = subset_idxs[j]
        curr_Y = np.take(Y, curr_idxs, axis=0)

        if j > 0:
            # orthogonalization step
            Q = gram_schmidt(curr_Y[:, compID])
            Rj = np.zeros(curr_Y.shape)
            for i in range(0, curr_Y.shape[1]):
                rj = np.matmul(np.matmul(Q, Q.T), curr_Y[:,i])
                Rj[:,i] = rj
                curr_Y = np.subtract(curr_Y, Rj) 
        
        # column vector containing variance for each column
        VT =  VT + np.sum(np.var(curr_Y, axis=0))

        # calculate scores of 1st pc for curr_Y using nipals algorithm
        t1 = pca_first_nipals(curr_Y)

        # maximise efs
        for i in range(0, curr_Y.shape[1]):
            x = np.atleast_2d(curr_Y[:, i]).T
            EFS[i] = np.divide( np.square(np.matmul(x.T, t1)), np.matmul(x.T,x) + np.finfo(float).eps)

        # select variable most correlated with first pc
        EFS[compID] = np.nan
        idx = np.nanargmax(EFS)

        # variance explained using matrix deflation
        x = np.atleast_2d(curr_Y[:,idx]).T
        th = np.matmul(np.linalg.pinv(x), curr_Y)
        Yhat = np.matmul(x, th)
        YhatP =  YhatP + Yhat

        # accumulated variance explained
        VEX = np.sum(np.var(YhatP, axis=0)) / VT * 100

        # store results
        S.append(np.asarray(Y[:,idx]))
        compID.append(idx)
        VarEx.append(VEX)

    return S, VarEx, compID