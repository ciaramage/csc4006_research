import numpy as np

def read_matrix(filename):
    mat = np.loadtxt(filename, delimiter=',')
    return mat

def write_matrix(filename, mat):
    np.savetxt(fname=filename, X=mat, delimiter=',' )