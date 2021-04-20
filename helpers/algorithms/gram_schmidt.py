# function to compute qr factorization of a matrix
# for orthogonalization
from numpy.linalg import qr

def gram_schmidt(A):    
    """[summary]

    Args:
        A (a 2D numpy array): 

    Returns:
        Q: A matrix whose columns are orthonormal columns
    """
    Q, _ = qr(A)
    return Q

