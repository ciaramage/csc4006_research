import numpy as np
from numpy.linalg import norm

def read_matrix_from_file(filename):
    mat = np.loadtxt(filename, delimiter=',')
    return mat

def write_matrix_to_file(filename, mat):
    np.savetxt(fname=filename, X=mat, delimiter=',' )


def get_matrix(mat_size:list):
    # generate 2D numpy array of size mat_size and populate with random samples from a 
    # uniform distribution over [0 and 1]
    mat = np.random.rand(mat_size[0], mat_size[1])
    return mat

def pca_first_nipals(X):
    """ Principal Component Analysis, implemented using NIPALS algorithm

    Args:
        X (2d numpy array): Matrix to find the first principal component of - using the NIPALS algorithm

    Returns:
        scores: The pca scores of first principal component calculated using NIPALS
    """
    tolerance = 1e-6 # tolerance for confergence
    v = X.shape[1] -1 # maximum column index
    idx = 0 # initial column index
    scores = []

    # select the first non-zero column of X as initial guess for scores
    x = np.atleast_2d(X[:,idx]).T
    
    while np.matmul(x.T, x) < 5*np.finfo(float).eps and idx < v:
        idx = idx+1
        x = np.atleast_2d(X[:,idx]).T
    
    if np.matmul(x.T, x) < 5*np.finfo(float).eps:
        print('***Warning... data has no non-zero columns') # all column variance  = 0
        return 
    
    scores = x 
    p = np.matmul(X.T, scores)
    p = np.divide(p, norm(p))
    tnew = np.matmul(X,p)

    # calculate largest component of x
    while norm(tnew-scores) > tolerance:
        scores = tnew
        p = np.matmul(X.T, scores)
        p = np.divide(p, norm(p))
        tnew = np.matmul(X,p)

    return scores

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
