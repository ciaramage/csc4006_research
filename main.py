def main():
    print("main started")
    ################################################################
    # Uncomment to run the realDataInfo function
    ################################################################
    #realDataInfo()

    ################################################################
    # Uncomment to run the randomDataInfo function
    ################################################################
    #randomDataInfo()

    ################################################################
    # Uncomment to run the plotRandomTimes function
    ################################################################
    #plot_random_times()

    ################################################################
    # Uncomment to run the all of the feature selection algorithms
    # on all of the real use case datasets 
    ################################################################
    """ for ds in datasets:
        plot_glg_real('glg', ds)
        
        plot_sg_real('sg',ds, 0.5)

        plot_real_sg_compare(ds, [0.3, 0.7])
     """

    ################################################################
    # Uncomment to run the grazy and lazy greedy feature selection
    # functions on individual datasets
    ################################################################
    # plot_glg_real('glg', 'Xpitprops')
    # plot_glg_real('glg', 'X50sites')
    # plot_glg_real('glg', 'breastCancerDiagnosis')
    # plot_glg_real('glg', 'frogs')

    ################################################################
    # Uncomment to run the stochastic greedy feature selection
    # functions on individual datasets randomly sampling 50% of the 
    # data each iteration. The percentage is given as a decimal and
    # can be changed - however the default is 50%.
    ################################################################
    # plot_sg_real('sg','Xpitprops', 0.5)
    # plot_sg_real('sg', 'X50sites', 0.5)
    # plot_sg_real('sg','breastCancerDiagnosis', 0.5)
    # plot_sg_real('sg','frogs', 0.5)

    ################################################################
    # Uncomment to run the stochastic greedy feature selection
    # functions on individual datasets and comparing randomly sampling 
    # 30% and 70% data each iteration. There is not default list of percentages,
    # the values given in the experiments are 30% and 70% - but they can be changed.
    ################################################################
    # plot_real_sg_compare('Xpitprops', [0.3, 0.7])
    # plot_real_sg_compare('X50sites', [0.3, 0.7])
    # plot_real_sg_compare('breastCancerDiagnosis',[0.3, 0.7])
    # plot_real_sg_compare('frogs', [0.3, 0.7])



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
