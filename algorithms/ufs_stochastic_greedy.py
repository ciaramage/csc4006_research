import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing

def ufs_stochastic_greedy(X, Nc, percentage=0.5):
    """This function implements Unsupervised Feature Selection algorithm with
    Stochastic Greedy (also known as lazier than lazy greedy) optimisation applied.
    At each iteration a random sample of the original data is taken and is used to 
    select the next feature. For more than two components, the indices of features selected
    in previous iterations are used to orthogonalise the features in the current subset using
    the Gram-Schmidt process.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        percentage (int, optional): The percentage of data selected from the original data. Defaults to 0.5

    Returns:
        S: The column vectors of each selected feature during each iteration
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
    rSquare.append(np.min(sq_corr, axis=1)[c_idxs])
    
    # Loop for remaining components
    for _ in range(0, Nc-2):
        # Update sample
        y_idx = get_sample_idxs()
        Y = np.take(X, y_idx, axis=0)

        # Choose an orthonormal basis c = {c1,c2} for the subspace of R^P spanned by the first two columns
        # if the first two columns are Xa and Xb
        # c1 = Xa, and c2 = Z/|Z| - where Z = Xa - (Xb*Xa)*Xa
        c = get_c(Y, compID)

        # If > 2 components have already been selected
        if len(compID) > 2:
            # Update the orthonormal basis for the subspace spanned by the selected columns: c
            c = get_c(Y, compID)
            M = np.append(M, c)
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

    return S, M, rSquare.tolist(), compID.tolist()

def get_c(X, idxs):
    c = np.atleast_2d(X[:,idxs[0]]).T
    for i in range(1, len(idxs)):
        Xi = np.atleast_2d(X[:,idxs[i]]).T
        ci = Xi - np.matmul(np.matmul(c, c.T), Xi)
        ci = np.divide(ci, norm(ci))
        c = np.append(c, ci, axis=1)
    return c