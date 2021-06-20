import time
import numpy as np


def main():
    print("main started")
    #realDataInfo()
    Nc1 = 8
    #mat = np.loadtxt('data/X50sites.txt', delimiter=',')

    """"     sizes = [(500, 50), (500, 100), (500,150), (500,200), (500,250),(500,300), (500,350), (500,400), (500,450), (500,500)]
    
    for i in range(len(sizes)): """
    results = []
    mat = read_matrix('data/realData/forestFires.txt')

    start_fsca = time.time()
    _, _, varEx_fsca, compID_fsca = UFS(mat, Nc)
    duration_fsca = time.time() - start_fsca
    results.append(['ufs', varEx_fsca, compID_fsca, duration_fsca])

    for result in results:
        print(result)
    
    get_random_duration()

    """ start_ufs = time.time()
    _, _, varEx_ufs, compID_ufs = lazy_greedy_UFS(mat, 8)
    duration_ufs = time.time() - start_ufs
    results.append(['ufs lg', varEx_ufs, compID_ufs, duration_ufs])

    start_fsca_lg = time.time()
    _,_, varEx_fsca_lg, compID_fsca_lg = stochastic_ufs(mat, 8, 0.3)
    duration_fsca_lg = time.time() - start_fsca_lg
    results.append(['ufs sg', varEx_fsca_lg, compID_fsca_lg, duration_fsca_lg]) """

    """ for i in range(len(sizes)):
        mat = get_matrix(sizes[i])
        write_matrix('data/randomData/t{0}.txt'.format(i+1), mat)

    randomDataInfo() """

if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from helpers.matrix_in_out import *
        from helpers.matrix_generator import get_matrix
        #from helpers.toLatex import realDataInfo, randomDataInfo
        from algorithms.fsca import fsca
        from algorithms.opfs import opfs
        from algorithms.ufs import ufs
        from algorithms.fsca_lazy_greedy import fsca_lazy_greedy
        from algorithms.ufs_lazy_greedy import ufs_lazy_greedy
        from algorithms.fsca_stochastic_greedy import fsca_stochastic_greedy_orthogonal    
        from algorithms.fsca_stochastic_def import fsca_stochastic_greedy_deflation
        from algorithms.opfs_stochastic_greedy import opfs_stochastic_greedy_orthogonal   
        from algorithms.opfs_stochastic_greedy_deflation import opfs_stochastic_greedy_deflation
        from algorithms.ufs_stochastic_greedy import ufs_stochastic_greedy
    else:
        from .algorithms.ufs import UFS
        from .algorithms.fsca import FSCA
        from .algorithms.opfs import OPFS
        from .algorithms.fsca_lazy_greedy import lazy_greedy_FSCA
        from .algorithms.ufs_lazy_greedy import UFS_new
        from .algorithms.fsca_stochastic_greedy import STOCHASTIC_FSCA_ORT
        from .algorithms.fsca_stochastic_def import STOCHASTIC_FSCA_DEF
        from .algorithms.opfs_stochastic_greedy import STOCHASTIC_OPFS
        from .algorithms.opfs_stochastic_greedy_deflation import STOCHASTIC_OPFS_DEF
        from .algorithms.ufs_stochastic_greedy import STOCHASTIC_UFS
    main()
