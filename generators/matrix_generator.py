import numpy as np
from sklearn import preprocessing
from ..helpers.MatrixTypes import MatrixTypes

def independent_mat(mat_size:list):
    # generate matrix with random values between 0 and 1
    mat = np.random.rand(mat_size[0], mat_size[1])
    return mat

def zero_mean_mat(mat_size:list):
    # generate matrix with random values between 0 and 1
    mat = np.random.rand(mat_size[0], mat_size[1])
    # scale to zero mean but not unit variance
    mat = preprocessing.scale(mat, with_mean=True, with_std=False)
    return mat
    
def partially_independent_mat(mat_size:list):
    # TODO

def get_matrix(mat_size:list, mat_type:MatrixTypes):
    if mat_type == MatrixTypes.INDEPENDENT:
        return independent_mat(mat_size)
    elif mat_type == MatrixTypes.ZERO_MEAN:
        return zero_mean_mat(mat_size)
    elif mat_type == MatrixTypes.PARTIALLY_INDEPENDENT:
        return partially_independent_mat(mat_size)
    elif mat_type == MatrixTypes.INDEPENDENT_CORR:
        mat = independent_correlation_mat(mat_size)
        return np.corrcoef(mat)
    elif mat_type == MatrixTypes.ZERO_MEAN_CORR:
        mat = zero_mean_correlation_mat(mat_size)
        return np.corrcoef(mat)
    else:
        mat = partially_independent_correlation_mat(mat_size)
        return np.corrcoef(mat)

    