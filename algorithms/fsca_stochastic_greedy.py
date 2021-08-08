import numpy as np
from numpy.linalg import qr

def fsca_stochastic_greedy_orthogonal( X, Nc, percentage=0.5):
    """ This function implements the Forward Selection Component Analysis algorithm with
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

    # Initialise storage for rayleigh quotient values
    rQ = np.zeros(Y.shape[1])

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
    VEX = VEX + np.divide(100*v, TR)

    # Store first component results
    compID.append(idx)
    VarEx.append(VEX)

    # Loop for remaining components
    for i in range(1,Nc):
        # Update sample
        idxs = get_sample_idxs()
        Y = np.take(X, idxs, axis=0)

        TR = np.trace(np.matmul(Y.T, Y))

        # Orthogonalise the current subset using the already selected columns as a basis
        #Q, _ = qr(Y[:,compID]) # Columns of Q are orthonormal
        Q = gram_schmidt(Y[:,compID])
        Yj = np.zeros(Y.shape)
        for j in range(L):
            yj = np.matmul(np.matmul(Q, Q.T), Y[:,j])
            Yj[:,j] = yj
        Y = np.subtract(Y, Yj)
        M.append(Q)

        # Calculate the Rayleigh Quotients
        # Update storage for rayleigh quotient values
        rQ = np.zeros(Y.shape[1])
        for j in range(Y.shape[1]):
            # Column j
            x = np.atleast_2d(Y[:,j]).T
            # Rayleigh quotient for column 
            r = np.matmul(Y.T, x)
            rQ[j] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))

        # Maximise Rayleigh Quotients
        idx = np.nanargmax(rQ)
        v = rQ[idx]

        # Calculate the accumulated variance explained with the inclusion of this feature
        VEX = VEX + np.divide(100*v, TR)
        
        # Store results
        compID.append(idx)
        VarEx.append(VEX)
        
    S = X[:,compID]

    return S, M, VarEx, compID

def gs(A):
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:,k], A[:,k]))
        Q[:,k] = A[:,k] / R[k,k]
        for j in range(k+1, A.shape[1]):
            R[k,j] = np.dot(Q[:,k], A[:,j])
            A[:,j] = A[:,j] - R[k,j] * Q[:,k]
    return Q

def gram_schmidt(X):
    """ Implements Gram-Schmidt orthogonalization.

    Args:
        X (A 2D numpy array): The columns of X are linearly independent

    Returns:
        U: (A 2D numpy array): The column of U are orthonormal
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