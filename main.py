import time
import numpy as np
import matplotlib.pyplot as plt

# write results of pca - 1 component selection to file
def write_pca_analysis(mat_size, pca_duration):
    string_to_write = str(mat_size) + "," + str(pca_duration)
    return string_to_write


def main():
    print("main started")
    m = (10,30)
    #Nc = 8
    #mat = get_matrix(m, MatrixTypes.INDEPENDENT)
    #np.savetxt('test.out', mat, delimiter=',')

    mat = np.loadtxt('test.out', delimiter=',')

    S, M, smallestSquare, compID = do_ufs(mat, 5)
    

if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from generators.matrix_generator import get_matrix
        from helpers.algorithms.pca import pca_nipals
        from helpers.MatrixTypes import MatrixTypes
        from algorithms.fsca import do_fsca
        from algorithms.opfs import do_opfs
        from algorithms.ufs import do_ufs
        from helpers.algorithms.gram_schmidt import gram_schmidt
    else:
        from .generators.matrix_generator import get_matrix
        from ..helpers.algorithms.pca import pca_nipals
        from ..helpers.MatrixTypes import MatrixTypes
        from ..helpers.algorithms.gram_schmidt import gram_schmidt
        from .algorithms.ufs import do_ufs
        from .algorithms.fsca import do_fsca
        from .algorithms.opfs import do_opfs
    main()
