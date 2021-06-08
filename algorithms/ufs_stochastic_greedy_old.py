import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing

def stochastic_ufs(X, Nc, percentage=0.9):
    """ This function implements the Unsupervised Feature Selection algorithm with
    Stochastic Greedy (also known as lazier than lazy greedy) optimisation applied.
    At each iteration a random sample of the original data is taken and is used to 
    select the next feature.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        with_replacement (logical boolean, optiona;): Dictates whether random sampling
            should occur with or without replacement.
        percentage (int, optional): If random sampling occurs with replacement - this is the percentage
            of data selected from the original data. Defaults to 0.1

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: 
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features 
    """
    # Normalise matrix columns to have zero mean and unit variance
    X = preprocessing.normalize(X, axis=0)
    sample_size = int(X.shape[0] * percentage)

    def get_sample_idxs():
        return np.random.randint(low = 0, high = X.shape[1], size = sample_size, dtype=int)

    # Subset for selecting first two features
    idxs = get_sample_idxs()
    Y = X[idxs,:]

    sq_corr = np.square(np.matmul(Y.T, Y))
    c_idxs = np.argpartition(np.min(sq_corr, axis=1), kth=1)[:2]

    col_idxs = np.arange(X.shape[1])

    S = []
    M = []
    rSquare = []
    compID = []

    c1 = np.atleast_2d(Y[:,c_idxs[0]]).T
    Xb = np.atleast_2d(Y[:,c_idxs[1]]).T
    c2 = c1 - np.dot(c1.T, Xb) * c1
    c2 = np.divide(c2, norm(c2))
    c = np.append(c1, c2, axis=1)

    compID = c_idxs
    rSquare.append(np.min(sq_corr, axis=1)[c_idxs])

    col_idxs = np.delete(col_idxs, c_idxs)

    # loop for remaining components
    for i in range(2, Nc):
        # Update sample
        idxs = get_sample_idxs()
        Y = X[idxs,:]

        R = norm(np.matmul(np.matmul(c, c.T), Y[:,col_idxs]), axis=0)
        idx = np.argmin(R)
        v = R[idx]

        compID = np.append(compID, idx)
        rSquare = np.append(rSquare, v)

        Xj = np.atleast_2d(Y[:,col_idxs[idx]]).T
        ck = Xj - np.matmul(np.matmul(c, c.T), Xj)
        ck = np.divide(ck, norm(ck))

        c = np.append(c, ck, axis=1)

        col_idxs = np.delete(col_idxs, idx)

    S = X[:,compID]
    M = c

    return S, M, rSquare, compID
