import time
import numpy as np


def main():
    print("main started")
    #realDataInfo()
    Nc = 5
    #mat = np.loadtxt('data/X50sites.txt', delimiter=',')

    """"     sizes = [(500, 50), (500, 100), (500,150), (500,200), (500,250),(500,300), (500,350), (500,400), (500,450), (500,500)]
    
    for i in range(len(sizes)): """
    results = []
    mat = read_matrix('data/realData/forestFires.txt')

    start_fsca = time.time()
    _, _, varEx_fsca, compID_fsca = UFS(mat, 8)
    duration_fsca = time.time() - start_fsca
    results.append(['ufs', varEx_fsca, compID_fsca, duration_fsca])

    start_ufs = time.time()
    _, _, varEx_ufs, compID_ufs = lazy_greedy_UFS(mat, 8)
    duration_ufs = time.time() - start_ufs
    results.append(['ufs lg', varEx_ufs, compID_ufs, duration_ufs])

    start_fsca_lg = time.time()
    _,_, varEx_fsca_lg, compID_fsca_lg = stochastic_ufs(mat, 8, 0.3)
    duration_fsca_lg = time.time() - start_fsca_lg
    results.append(['ufs sg', varEx_fsca_lg, compID_fsca_lg, duration_fsca_lg])

    for result in results:
        print(result)

    """ for i in range(len(sizes)):
        mat = get_matrix(sizes[i])
        write_matrix('data/randomData/t{0}.txt'.format(i+1), mat)

    randomDataInfo() """
    
    """     results1 = []
    print('\n4nc fsca')
    

    print('\n4nc ufs')
    

    print('\n4nc lg fsca')
     """

    """ results = [];
    sizes = [(500, 50), (500, 100), (500,150), (500,200), (500,250),(500,300), (500,350), (500,400), (500,450), (500,500)]"""

    


    """start_fsca = time.time()
    S_fsca, M_fsca, varEx_fsca, compID_fsca = do_fsca(mat, Nc)
    duration_fsca = time.time() - start_fsca """

    """ start_fsca1 = time.time()
    S_fsca1, M, varEx_fsca1, compID_fsca1 = do_ufs_lazy_greedy(mat, Nc)
    duration_fsca1 = time.time() - start_fsca1"""
    
    """start_fsca2 = time.time()
    S_fsca2, varEx_fsca2, compID_fsca2 = STOCHASTIC_UFS(mat, Nc, 0.99, True, 0.3)
    duration_fsca2 = time.time() - start_fsca2 """


    """  print('\n\n')
    print(varEx_fsca, compID_fsca)
    print(duration_fsca)


    print('\n\n')
    print(varEx_fsca2, compID_fsca2)
    print(duration_fsca2) """




if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from helpers.matrix_in_out import *
        from helpers.matrix_generator import get_matrix
        from helpers.algorithms.pca import pca_first_nipals
        from helpers.dataStructures.PriorityQueue import PriorityQueue
        from helpers.toLatex import *
        from algorithms.fsca import FSCA
        from algorithms.opfs import OPFS
        from algorithms.ufs import UFS
        from algorithms.fsca_lazy_greedy import lazy_greedy_FSCA
        from algorithms.fsca_lazy_greedy_PQ import lazy_greedy_FSCA_PQ
        from algorithms.ufs_lazy_greedy import lazy_greedy_UFS
        from algorithms.fsca_stochastic_greedy import STOCHASTIC_FSCA_ORT    
        from algorithms.fsca_stochastic_greedy_deflation import STOCHASTIC_FSCA_DEF
        from algorithms.opfs_stochastic_greedy import STOCHASTIC_OPFS   
        from algorithms.opfs_stochastic_greedy_deflation import STOCHASTIC_OPFS_DEF
        from algorithms.ufs_stochastic_greedy import stochastic_ufs
    else:
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
        from .algorithms.ufs_stochastic_greedy import STOCHASTIC_UFS
    main()
