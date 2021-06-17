import time
import numpy as np


def main():
    print("main started")
    #realDataInfo()
    Nc1 = 8
    #mat = np.loadtxt('data/X50sites.txt', delimiter=',')

    """"     sizes = [(500, 50), (500, 100), (500,150), (500,200), (500,250),(500,300), (500,350), (500,400), (500,450), (500,500)]
    
    for i in range(len(sizes)): """
    results1 = []
    mat = read_matrix('data/realData/X50sites.txt')

    start_ufs = time.time()
    _, _, varEx_ufs, compID_ufs = ufs(mat, Nc1)
    duration_ufs = time.time() - start_ufs
    results1.append(['UFS', varEx_ufs, compID_ufs, duration_ufs]) 

    print('\n4nc lg ufs')
    start_ufs_lg = time.time()
    _, _, varEx_ufs_lg, compID_ufs_lg = ufs_lazy_greedy(mat, Nc1)
    duration_ufs_lg = time.time() - start_ufs_lg
    results1.append(['OPFS SG DEF', varEx_ufs_lg, compID_ufs_lg, duration_ufs_lg])

    print('\n4nc sg ufs')
    start_ufs_sg = time.time()
    _, _, rSquare_ufs_sg, compID_ufs_sg = ufs_stochastic_greedy(mat, Nc1, 0.4)
    duration_ufs_sg = time.time() - start_ufs_sg
    results1.append(['OPFS SG DEF OLD', rSquare_ufs_sg, compID_ufs_sg, duration_ufs_sg])

    """ print('\n4nc sg ufs')
    start_ufs_sg1 = time.time()
    _, _,rSquare_ufs_sg1, compID_ufs_sg1 = stochastic_ufs(mat, Nc1, 0.1)
    duration_ufs_sg1 = time.time() - start_ufs_sg1
    results1.append(['FSCA SG ORT', rSquare_ufs_sg1, compID_ufs_sg1, duration_ufs_sg1]) """
 

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
        write_matrix('data/randomData/t{0}.txt'.format(i+1), mat)

    randomDataInfo() """
    
    """     results1 = []
    print('\n4nc fsca')
    

    print('\n4nc ufs')
    

    print('\n4nc lg fsca')
     """

    """ results = [];
    sizes = [(500, 50), (500, 100), (500,150), (500,200), (500,250),(500,300), (500,350), (500,400), (500,450), (500,500)]"""


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
