import numpy as np
from numpy.linalg import qr

def norm(u):
    """The funciton returns the magintude of input vector u

    Args:
        u (array-like): Vector to calculate magnitude of

    Returns:
        Number: The magnitude of vector u which is given as: ||u|| 
    """
    return np.sqrt(np.sum(u * u))

def project(u, v):
    """This function projects vector v onto vector u

    Args:
        u (array-like): Vector to project on
        v (array-like): Vector to project

    Returns:
        (array-like): The projection of vector v onto vector u
    """
    return np.matmul(np.divide(np.matmul(v,u),np.square(u)), u)

def gram_schmidt(A):    
    """ numpy.linalg.qr computes the qr factorization of a matrix
    The matrix a is factored as q, _m where q is orthonormal and _ is
    the upper triangular but is not used by this function.

    Args:
        A (a 2D numpy array representing a matrix): 

    Returns:
        Q: A matrix whose columns are orthonormal columns
    """
    Q, _ = qr(A)
    return Q

def gramSchmidt_process(A):
    """[summary]

    Args:
        A ([type]): [description]

    Returns:
        [type]: [description]
    """
    v = A.shape[1]

    for j in range(v):
        for k in range(j):
            A[:,j] -= np.dot(A[:,k], A[:,j]) * A[:,k]
        A[:,j] = A[:,j] / np.linalg.norm(A[:,j])
    return A

def orthogonal_projection(X, V=None, I=None):
    """[summary]

    Args:
        X ([type]): [description]
        V ([type], optional): [description]. Defaults to None.
        I ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    if(I is not None):
        V = np.take(X, I, axis=1)
    orthoV = gramSchmidt_process(V)

    Rj = np.zeros(X.shape)
    for i in range(0, X.shape[1]):
        rj = np.matmul(np.matmul(orthoV, orthoV.T), X[:,i])
        Rj[:,i] = rj
    return Rj
    

