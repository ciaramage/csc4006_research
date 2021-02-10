from enum import Enum

class MatrixTypes(Enum):
    INDEPENDENT = 1 # linearly independent
    ZERO_MEAN = 2 # zero mean columns
    FUNCTION_AS_MATRIX = 3 # each element is dependent on its row and column index
    ZERO_MEAN_CORR = 4 # correlation matrix from zero mean columns
    INDEPENDENT_CORR =5 # correlation matrix from independent
