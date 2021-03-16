# function to compute qr factorization of a matrix
# for orthogonalization
from numpy.linalg import qr
import numpy as np

# idx1: index of first coloumn
# idx2: index
def gram_schmidt(mat, cols):    
    # matrix containing the columns specified by the (starting column) idx1 and (ending column - inclusive) idx2
    A = mat[:,cols]
    Q, R = qr(A)
    return Q

