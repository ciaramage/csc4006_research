import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("main started")

    Nc = 5
    mat = np.loadtxt('data/Xpitprops.txt', delimiter=',')

    start_fsca = time.time()
    S, M, varEx, compID = do_fsca(mat, 10)
    duration_fsca = time.time() - start_fsca

    start_lazy = time.time()
    S_fsca, M_fsca, varEx_fsca, compID_fsca = do_lazy_fsca(mat, 10)
    duration_lazy = time.time() - start_lazy

    print(varEx, compID)
    print(duration_fsca)
    print('\n\n')
    print(varEx_fsca, compID_fsca)
    print(duration_lazy)

if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from generators.matrix_generator import get_matrix
        from helpers.algorithms.pca import pca_first_nipals
        from helpers.MatrixTypes import MatrixTypes
        from algorithms.fsca import do_fsca
        from algorithms.opfs import do_opfs
        from algorithms.ufs import do_ufs
        from algorithms.fsca_lazy_greedy import do_lazy_fsca
        from helpers.algorithms.gram_schmidt import gram_schmidt
    else:
        from .generators.matrix_generator import get_matrix
        from ..helpers.algorithms.pca import pca_nipals
        from ..helpers.MatrixTypes import MatrixTypes
        from ..helpers.algorithms.gram_schmidt import gram_schmidt
        from .algorithms.ufs import do_ufs
        from .algorithms.fsca import do_fsca
        from .algorithms.opfs import do_opfs
        from .algorithms.fsca_lazy_greedy import do_lazy_fsca
    main()
