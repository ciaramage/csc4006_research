import time

# write results of pca - 1 component selection to file
def write_pca_analysis(mat_size, pca_duration):
    string_to_write = str(mat_size) + "," + str(pca_duration)
    return string_to_write


def main():
    print("main started")

    start_time = time.time()
    # do pca analysis
    square_mat = get_matrix(True)
    # not_square_mat = get_matrix(False)

    
    #pca_results = []
    S, M, VarEx, compId = do_opfs(square_mat[6])
    print(S)
    print(VarEx)
    print(M)
    print(compId)
      
    duration = time.time() - start_time
    #print(results)
    print(duration)
    


if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from generators.matrix_generator import get_matrix
        from helpers.algorithms.pca import pca_nipals
        from algorithms.fsca import do_fsca
        from algorithms.opfs import do_opfs
    else:
        from .generators.matrix_generator import get_matrix
        from ..helpers.algorithms.pca import pca_nipals
        from .algorithms.fsca import do_fsca
        from .algorithms.opfs import do_opfs
    main()
