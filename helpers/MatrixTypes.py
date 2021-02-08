from enum import Enum

class MatrixTypes(Enum):
    INDEPENDENT = '1' # linearly independent
    ZERO_MEAN = '2' # zero mean columns
    CORRELATION = '3' #correlation matrix
    PARTIALLY_INDEPENDENT = '4' # first columns independent - the other columns are functions of the first
