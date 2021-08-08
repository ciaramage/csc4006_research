import numpy as np 
from helpers.pca import pca_first_nipals
from numpy.linalg import norm

def opfs_stochastic_greedy_orthogonal(X, Nc=1, percentage=0.5):
    """This function implements the Orthogonal Principal Feature Selection algorithm with
    Stochastic Greedy (also known as lazier than lazy greedy) optimisation applied.
    At each iteration a random sample of the original data is taken and is used to 
    select the next feature. For more than one component, the indices of features selected
    in previous iterations are used to orthogonalise the features in the current subset using
    the Gram-Schmidt process.

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
    # Matrix needs to be zero mean
    mX = X.mean(axis=0)
    if(max(mX) > 10**-6):
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX

    sample_size = int(X.shape[0] * percentage)

    def get_sample_idxs():
        """Returns a random sample of the row indices in data matrix X

        Returns:
            list: A random sample of the row indices in X
        """
        return np.random.randint(low = 0, high = X.shape[1], size = sample_size, dtype=int)

    # Initialize storage variables
    S = []
    M = []
    varEx = []
    compID = []
    vex = 0
    VT = 0

    # Data subset for this iteration
    y_idxs = get_sample_idxs()
    Y = np.take(X, y_idxs, axis=0)

    # Initialise storage for eigen vectors
    EFS = np.zeros(Y.shape[1])

    ## First component
    # Column vector containing variance for each column
    VT =  np.var(Y)

    # Calculate scores of 1st pc for curr_Y using nipals algorithm
    pc = pca_first_nipals(Y)

    # Maximise efs
    for i in range(Y.shape[1]):
        # feature column f in Y
        f = np.atleast_2d(Y[:,i]).T
        corr = np.divide( np.matmul(f.T, pc), norm(f))
        EFS[i] = corr

    # Select variable most correlated with first pc
    idx = np.nanargmax(EFS)
    x = np.atleast_2d(Y[:,idx]).T

    # Variance explained using matrix deflation
    P = gram_schmidt(x) # P -> subspace spanned by columns of Y
    #Yj = P @ P.T @ x # project feature X onto P

    # For each remaining column in Y, project to the subspace orthogonal to feature x
    Yj = np.empty(Y.shape)
    for i in range(Y.shape[1]):
        yj = np.matmul(np.matmul(P, P.T), Y[:,i])
        Yj[:,i] = yj
    Y = np.subtract(Y, Yj) 
    vex =  vex + np.var(Yj)

    # Accumulated variance explained
    VEX =  np.divide(vex, VT) * 100

    # Store results
    compID.append(idx)
    varEx.append(VEX)

    # Update col_idxs bby removing index of selected feature
    #col_idxs = np.delete(col_idxs, idx)
    
    # Loop for remaining components
    for _ in range(1, Nc):
        EFS = np.zeros(Y.shape[1])

        # Data subset for this iteration
        y_idxs = get_sample_idxs()
        Y = np.take(X, y_idxs, axis=0)

        # Orthogonalization step
        P = gram_schmidt(Y[:, compID])
        Yj = np.zeros(Y.shape)
        for i in range(0, Y.shape[1]):
            yj = np.matmul(np.matmul(P, P.T), Y[:,i])
            Yj[:,i] = yj
        Y = np.subtract(Y, Yj) 
        M.append(P)

        # Calculate scores of 1st pc for remaining columns using nipals algorithm
        pc = pca_first_nipals(Y)

        # Maximise efs
        for i in range(Y.shape[1]):
                # feature column f in Y
            f = np.atleast_2d(Y[:,i]).T
            corr = np.divide( np.matmul(f.T, pc), norm(f))
            EFS[i] = corr

        # Select variable most correlated with first pc
        idx = np.nanargmax(EFS)
        x = np.atleast_2d(Y[:,idx]).T

        # Projection of vector x onto the subspace spanned by the columns of Y
        P = gram_schmidt(x) # P -> subspace spanned by columns of Y

        # For each column in Y, project to the subspace orthogonal to feature x
        Yj = np.empty(Y.shape)
        for i in range(Y.shape[1]):
            yj = np.matmul(np.matmul(P, P.T), Y[:,i])
            Yj[:,i] = yj
        vex =  vex + np.var(Yj)
        varEx.append(vex / VT * 100)

        # Store results
        compID.append(idx)

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