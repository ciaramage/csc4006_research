import numpy as np

def independent_mat(mat_size:list):
    # generate 2D numpy array of size mat_size with random values between 0 and 1
    mat = np.random.rand(mat_size[0], mat_size[1])
    return mat

def get_matrix(mat_size:list):
        return independent_mat(mat_size)



    