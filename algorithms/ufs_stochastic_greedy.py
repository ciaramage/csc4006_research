import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing

def stochastic_ufs(X, Nc, percentage=0.4):

    # Normalise matrix columns to have zero mean and unit variance
    X = preprocessing.normalize(X, axis=0)
    sample_size = int(X.shape[0] * percentage)

    def get_sample_idxs():
        return np.random.randint(low = 0, high = X.shape[1], size = sample_size, dtype=int)

    # Subset for selecting first two features
    idxs = get_sample_idxs()
    Y = np.take(X, idxs, axis=0)

    # Square correlation matrix: (X^T * X)**2 
    sq_corr = np.square(np.matmul(Y.T, Y))
    # Select as the first two columns those with the smallest squared correlation coefficient
    c_idxs = np.argpartition(np.min(sq_corr, axis=1), kth=1)[:2]
    # Keep track of column indexes not selected
    col_idxs = np.arange(X.shape[1])
    col_idxs = np.delete(col_idxs, c_idxs)

    # Setup storage variables
    S = []
    M = []
    rSquare = []
    compID = []
    compID = c_idxs
    # Loop for remaining components
    for i in range(0, Nc-2):
        # Update sample
        y_idx = get_sample_idxs()
        Y = np.take(X, y_idx, axis=0)
        print(Y.shape)
        # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the first two columns
        # if the first two columns are Xa and Xb
        # c1 = Xa, and c2 = Z/|Z| - where Z = Xa - (Xb*Xa)*Xa
        c1 = np.atleast_2d(Y[:,c_idxs[0]]).T
        Xb = np.atleast_2d(Y[:,c_idxs[1]]).T
        c2 = np.subtract(c1, np.dot(c1.T, Xb)*c1)
        c2 = np.divide(c2, norm(c2))
        c = np.append(c1, c2, axis=1)

        # If > 2 components have already been selected
        if len(compID) > 2:
            # Update the orthonormal basis for the subspace spanned by the selected columns: c
            for j in range(2, len(compID)):
                Xj = np.atleast_2d(Y[:,j]).T
                ck = Xj - np.matmul(np.matmul(c, c.T), Xj)
                ck = np.divide(ck, norm(ck))
                c = np.append(c, ck, axis=1)
        
        # For each remaining column, calculate its squared multiple correlation coefficient
        # R^2 with the selected columns
        R = norm(np.matmul(np.matmul(c, c.T), Y[:,col_idxs]), axis=0)
        idx = np.argmin(R)
        v = R[idx]

        compID = np.append(compID, col_idxs[idx])
        rSquare = np.append(rSquare, v)

        # Update col_idxs by removing the index of the column selected 
        # in the current iteration
        col_idxs = np.delete(col_idxs, idx)

        M = np.append(M, c)
    S = X[:,compID]

    return S, M, rSquare, compID