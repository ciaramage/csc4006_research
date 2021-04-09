import time
import numpy as np
import matplotlib.pyplot as plt
import heapq as  hq 
import itertools

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

    start_lazy_pq = time.time()
    S_pq, M_pq, varEx_pq, compID_pq = do_lazy_fsca_pq(mat,10)
    duration_lazy_pq = time.time() - start_lazy_pq
    
    print(varEx, compID)
    print(duration_fsca)
    print('\n\n')
    print(varEx_fsca, compID_fsca)
    print(duration_lazy)
    print('\n\n')
    print(varEx_pq, compID_pq)
    print(duration_lazy_pq)



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
        from helpers.algorithms.gram_schmidt import gram_schmidt
       
    else:
        from .generators.matrix_generator import get_matrix
        from ..helpers.algorithms.pca import pca_nipals
        from ..helpers.MatrixTypes import MatrixTypes
        from ..helpers.dataStructures.PriorityQueue import PriorityQueue
        from ..helpers.algorithms.gram_schmidt import gram_schmidt
        from .algorithms.ufs import do_ufs
        from .algorithms.fsca import do_fsca
        from .algorithms.opfs import do_opfs
        from .algorithms.fsca_lazy_greedy import do_lazy_fsca
        from .algorithms.fsca_lazy_greedy_PQ import do_lazy_fsca_pq
    main()
