import numpy as np
from numpy.linalg import norm
from helpers.pca import pca_first_nipals

def opfs(X, Nc=1):
    """This function implements the baseline Orthogonal Principal Feature Selection algorithm
    with no optimization applied.

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
    if( max(mX) > 10**-6):
        # Columns not mean centered
        print("\nWarning: Data not zero mean... detrending\n")
        X = X- mX    

    compID = []
    M = []
    varEx = []
    vex = 0
    VT =  np.var(Y)

    for _ in range(Nc):
        # Find the first eigenvector. Use PCA to find the principle component with the largest eigenvalue
        pc  = pca_first_nipals(Y)
        EFS = np.zeros(Y.shape[1])

        # Find the feature from the orthogonal dataset that is most correlated to the largest eigen vector
        for i in range(Y.shape[1]):
            # feature column f in Y
            f = np.atleast_2d(Y[:,i]).T
            corr = np.divide( np.matmul(f.T, pc), norm(f))
            EFS[i] = corr

        idx = np.nanargmax(EFS) # index of variable with max EFS
        x = np.atleast_2d(Y[:,idx]).T # feature of variable with max correlation with pc

        compID.append(idx)
        S = [x]

        # feature selected -> reduce the search space

        # Projection of vector x onto the subspace spanned by the columns of Y
        P = gram_schmidt(x) # P -> subspace spanned by columns of Y

        # For each remaining column in Y, project to the subspace orthogonal to feature x
        Yj = np.empty(Y.shape)
        for i in range(Y.shape[1]):
            yj = np.matmul(np.matmul(P, P.T), Y[:,i])
            Yj[:,i] = yj
        Y = np.subtract(Y, Yj) 
        vex =  vex + Yj

        varEx.append(np.var(vex) / VT * 100)

        M.append(P)

    S = X[:,compID]
    
    return S, M, varEx, compID


def gram_schmidt(X):
    """
    Implements Gram-Schmidt orthogonalization.

    Parameters
    ----------
    X : an n x k array with linearly independent columns

    Returns
    -------
    U : an n x k array with orthonormal columns

    """

    # Set up
    n, k = X.shape
    U = np.empty((n, k))
    I = np.eye(n)

    # The first col of U is just the normalized first col of X
    v1 = X[:,0]
    U[:, 0] = v1 / np.sqrt(np.sum(v1 * v1))

    for i in range(1, k):
        # Set up
        b = X[:, i]       # The vector we're going to project
        Z = X[:, 0:i]     # First i-1 columns of X

        # Project onto the orthogonal complement of the col span of Z
        M = I - Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        u = M @ b

        # Normalize
        U[:, i] = u / np.sqrt(np.sum(u * u))

    return U