import time
import numpy as np
import matplotlib.pyplot as plt
import heapq as  hq 
import itertools

def main():
    print("main started")

    Nc = 10
    mat = np.loadtxt('data/Xpitprops.txt', delimiter=',')
    

    start_fsca = time.time()
    S_fsca, M_fsca, varEx_fsca, compID_fsca = do_opfs(mat, Nc)
    duration_fsca = time.time() - start_fsca

    start_fsca1 = time.time()
    S_fsca1, varEx_fsca1, compID_fsca1 = STOCHASTIC_OPFS(mat, Nc, True, 0.3)
    duration_fsca1 = time.time() - start_fsca1
    
    start_fsca2 = time.time()
    S_fsca2, varEx_fsca2, compID_fsca2 = STOCHASTIC_OPFS_DEF(mat, Nc, True, 0.3)
    duration_fsca2 = time.time() - start_fsca2


    print('\n\n')
    print(varEx_fsca, compID_fsca)
    print(duration_fsca)

    print('\n\n')
    print(varEx_fsca1, compID_fsca1)
    print(duration_fsca1)

    print('\n\n')
    print(varEx_fsca2, compID_fsca2)
    print(duration_fsca2)




if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from generators.matrix_generator import get_matrix
        from helpers.algorithms.pca import pca_first_nipals
        from helpers.MatrixTypes import MatrixTypes
        from helpers.dataStructures.PriorityQueue import PriorityQueue
        from algorithms.fsca import do_fsca
        from algorithms.opfs import do_opfs
        from algorithms.ufs import do_ufs
        from algorithms.fsca_lazy_greedy import do_lazy_fsca
        from algorithms.fsca_lazy_greedy_PQ import do_lazy_fsca_pq
        from algorithms.ufs_lazy_greedy import do_ufs_lazy_greedy
        from algorithms.fsca_stochastic_greedy import STOCHASTIC_FSCA_ORT    
        from algorithms.fsca_stochastic_greedy_deflation import STOCHASTIC_FSCA_DEF
        from algorithms.opfs_stochastic_greedy import STOCHASTIC_OPFS   
        from algorithms.opfs_stochastic_greedy_deflation import STOCHASTIC_OPFS_DEF
    else:
        from .generators.matrix_generator import get_matrix
        from ..helpers.algorithms.pca import pca_nipals
        from ..helpers.MatrixTypes import MatrixTypes
        from ..helpers.dataStructures.PriorityQueue import PriorityQueue
        from .algorithms.ufs import do_ufs
        from .algorithms.fsca import do_fsca
        from .algorithms.opfs import do_opfs
        from .algorithms.fsca_lazy_greedy import do_lazy_fsca
        from .algorithms.fsca_lazy_greedy_PQ import do_lazy_fsca_pq
        from .algorithms.ufs_lazy_greedy import do_ufs_lazy_greedy
        from .algorithms.fsca_stochastic_greedy import STOCHASTIC_FSCA_ORT
        from .algorithms.fsca_stochastic_greedy_deflation import STOCHASTIC_FSCA_DEF
        from .algorithms.opfs_stochastic_greedy import STOCHASTIC_OPFS
        from .algorithms.opfs_stochastic_greedy_deflation import STOCHASTIC_OPFS_DEF
    main()
