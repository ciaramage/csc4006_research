import numpy as np


def main():
    print("main started")

    realDataInfo()
    randomDataInfo()
    plot_random_times()
    for ds in datasets:
        plot_glg_real('glg', ds)
        
        plot_sg_real('sg',ds, 0.5)

        plot_real_sg_compare(ds, [0.3, 0.7])

if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.abspath(__file__)))
        from helpers.common import *
        from helpers.toLatex import *
        from helpers.experiment_results import *
    else:
        from .helpers.experiment_results import *
        from .helpers.common import *
        from .helpers.toLatex import *
    main()
