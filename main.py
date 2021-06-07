import time
import numpy as np


def main():

    results1 = []
    print("main started")
    
    Nc1=8
    mat = read_matrix('data/realData/Xpitprops.txt')
    print('\n4nc ufs')

    start_ufs = time.time()
    _, _, varEx_ufs, compID_ufs = UFS(mat, Nc1)
    duration_ufs = time.time() - start_ufs
    results1.append(['Baseline UFS new', varEx_ufs, compID_ufs, duration_ufs]) 

    print('\n4nc lg ufs')
    start_ufs_lg = time.time()
    _, _, varEx_ufs_lg, compID_ufs_lg = lazy_greedy_UFS(mat, Nc1)
    duration_ufs_lg = time.time() - start_ufs_lg
    results1.append(['Lazy Greedy UFS', varEx_ufs_lg, compID_ufs_lg, duration_ufs_lg])

    print('\n4nc sg ufs')
    start_ufs_sg = time.time()
    _, _,rSquare_ufs_sg, compID_ufs_sg = STOCHASTIC_UFS(mat, Nc1)
    duration_ufs_sg = time.time() - start_ufs_sg
    results1.append(['Stochastic Greedy UFS', rSquare_ufs_sg, compID_ufs_sg, duration_ufs_sg])
 

    for result in results1:
            print(result)
            print('\n')

    """ realDataInfo()
    randomDataInfo()
    results = [];
    sizes = [(500,10), (500, 50), (500, 100), (500,150), (500,200), (500,250),(500,300), (500,350), (500,400), (500,450), (500,500)] """

    """for i in range(len(sizes)):
        mat = get_matrix(sizes[i])
        write_matrix('data/randomData/t{0}.txt'.format(i+1), mat) """

    """ for i in range(len(sizes)):
        mat = get_matrix(sizes[i])
        read_matrix('data/randomData/t{0}.txt'.format(i+1))
        start = time.time()
        _,_,VarEx, compID = FSCA(mat, 6)
        duration= time.time() - start
        results.append([mat.shape, VarEx, compID, duration])
        print('\n')
        print(mat.shape)
        print(duration) """








if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from helpers.matrix_in_out import read_matrix, write_matrix
        from helpers.matrix_generator import get_matrix
        #from helpers.toLatex import realDataInfo, randomDataInfo
        from helpers.dataStructures.PriorityQueue import PriorityQueue
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
        from algorithms.ufs_stochastic_greedy import STOCHASTIC_UFS
    else:
        from .generators.matrix_generator import get_matrix
        from ..helpers.algorithms.pca import pca_nipals
        from ..helpers.MatrixTypes import MatrixTypes
        from ..helpers.dataStructures.PriorityQueue import PriorityQueue
        from .algorithms.ufs import UFS
        from .algorithms.fsca import FSCA
        from .algorithms.opfs import OPFS
        from .algorithms.fsca_lazy_greedy import lazy_greedy_FSCA
        from .algorithms.fsca_lazy_greedy_PQ import lazy_greedy_FSCA_PQ
        from .algorithms.ufs_lazy_greedy import UFS_new
        from .algorithms.fsca_stochastic_greedy import STOCHASTIC_FSCA_ORT
        from .algorithms.fsca_stochastic_greedy_deflation import STOCHASTIC_FSCA_DEF
        from .algorithms.opfs_stochastic_greedy import STOCHASTIC_OPFS
        from .algorithms.opfs_stochastic_greedy_deflation import STOCHASTIC_OPFS_DEF
        from .algorithms.ufs_stochastic_greedy import STOCHASTIC_UFS
    main()
