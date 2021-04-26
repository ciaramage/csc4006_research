from numpy.linalg import qr

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

