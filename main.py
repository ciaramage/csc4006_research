import time

# write results of pca - 1 component selection to file
def write_pca_analysis(mat_size, pca_duration):
    string_to_write = str(mat_size) + "," + str(pca_duration)
    return string_to_write


def main():
    print("main started")

    
    # do pca analysis
    square_mat = get_matrix(True)
    not_square_mat = get_matrix(False)

    # comparison of fsca and opfs

    # square matrices
    #for i in range(start=0, len(square_mat)):

    start_time = time.time()

    S_fsca, M_fsca, VarEx_fsca, comp_Id_fsca = do_opfs(square_mat[4])
    print(VarEx_fsca)
    print(comp_Id_fsca)
    #    S_opfs, M_opfs, VarEx_opfs, compId_opfs = do_opfs(square_mat[i])

      
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
        from .helpers.algorithms.pca import pca_nipals
        from .algorithms.fsca import do_fsca
        from .algorithms.opfs import do_opfs
    main()
