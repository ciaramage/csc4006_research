import numpy as np
from sklearn import preprocessing

square_dimensions = [10, 20, 50, 100, 200, 300, 400, 500, 1000 ]
not_square_dimensions = [[10, 20, 30, 40, 50, 60, 70], [100, 200, 500, 1000, 2000]]

#def get_linearly_independent_matrix(mat, n, d):
#    r = np.linalg.matrix_rank(mat)
#    while r < d:  # check the matrix rank to verify if matrix has linearly independent column vectors
#        mat = np.random.rand(n, d)
#        r = np.linalg.matrix_rank(mat)
#    return mat


def square_matrix():
    matrices = []
    for i in range(len(square_dimensions)):
        print(str(square_dimensions[i]) + "\t")
        mat = np.random.rand(square_dimensions[i], square_dimensions[i])  # matrix with random values
        #mat = get_linearly_independent_matrix(mat, square_dimensions[i], square_dimensions[i])
        #mat = np.linalg.pinv(mat)  # Moore Penrose pseudo inverse of a matrix

        # standardize data attributes to have mean of zero and standard deviation of 1
        mat = preprocessing.scale(mat)
        matrices.append(mat)
    return matrices


def not_square_matrix():
    matrices = []
    for i in range(len(not_square_dimensions[0])):
        for j in range(len(not_square_dimensions[1])):
            mat = np.random.rand(not_square_dimensions[0][i], not_square_dimensions[1][j])
            #mat = get_linearly_independent_matrix(mat, not_square_dimensions[0][i], not_square_dimensions[1][j])
            # Moore Penrose pseudo inverse of a matrix
            #mat = np.linalg.pinv(mat)
            mat = preprocessing.scale(mat)
            matrices.append(mat)
            matrices.append(mat)
    return matrices


def get_matrix(is_square):
    if is_square:
        return square_matrix()
    else:
        return not_square_matrix()

