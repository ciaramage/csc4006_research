from enum import Enum

class MatrixTypes(Enum):
    INDEPENDENT = '1' # linearly independent
    ZERO_MEAN = '2' # zero mean columns
    PARTIALLY_INDEPENDENT = '4' # first columns independent - the other columns are functions of the first
    ZERO_MEAN_CORR = '5' # correlation matrix from zero mean columns
    INDEPENDENT_CORR ='6' # correlation matrix from independent
    PARTIALLY_INDEPENDENT_CORR = '7' # correlation matrix from partially independet