import numpy as np
import math
from sklearn import preprocessing
from helpers.MatrixTypes import MatrixTypes

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
    
def function_as_mat(mat_size:list):
    # each element is dependent on its row and column index
    # element wise ->  row_idx * cosine(col_idx)
    mat = np.fromfunction(lambda x, y : x*np.cos(y), mat_size, dtype=float)
    # replace 0 value with 0.1
    #mat = np.where(mat==0., 0.1, mat) 
    return mat


def get_matrix(mat_size:list, mat_type:MatrixTypes):
    if mat_type == MatrixTypes.INDEPENDENT:
        return independent_mat(mat_size)
    elif mat_type == MatrixTypes.ZERO_MEAN:
        return zero_mean_mat(mat_size)
    elif mat_type == MatrixTypes.FUNCTION_AS_MATRIX:
        return function_as_mat(mat_size)
    elif mat_type == MatrixTypes.INDEPENDENT_CORR:
        mat = independent_mat(mat_size)
        return np.corrcoef(mat)
    else:
        mat = zero_mean_mat(mat_size)
        return np.corrcoef(mat)


    