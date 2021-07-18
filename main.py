import time
import numpy as np


def main():
    print("main started")
    #x, d = random_duration('fsca')
    #plot_random_times()
    #randomDataInfo()
    #realDataInfo()

    #plot_real('glg','Xpitprops')

    real_results('opfs', 'breastCancerDiagnosis', 0.5)

    #real_sg_compare_results('Xpitprops', [0.1,0.3,0.5,0.7,0.9])

    #plot_real_sg_compare('Xpitprops', [0.1,0.3,0.5,0.7,0.9])

    """ sizes = [(50, 20), (100,20), (150,20), (200,20), (250,20), (300,20), (350,20), (400, 20), (450,20), (500,20)]

    for i in range(len(sizes)):
        mat = get_matrix(sizes[i])
        write_matrix('data/randomData/t{0}.txt'.format(i+1), mat) """


if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from helpers.matrix_in_out import *
        #from helpers.matrix_generator import get_matrix
        from helpers.toLatex import *
        from helpers.experiment_results import *
    else:
        from .helpers.experiment_results import *

    main()
