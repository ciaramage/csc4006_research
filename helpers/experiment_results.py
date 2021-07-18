import numpy as np
from collections import defaultdict
from helpers.feature_selection import Result, select_features
from helpers.matrix_in_out import read_matrix

algorithm_names = [ 'FSCA', 'LG-FSCA', 'SGO-FSCA', 'SGD-FSCA', 'OPFS', 'SGO-OPFS', 'SGD-OPFS', 'UFS', 'LG-UFS', 'SG-UFS']
fsca_names = [ 'FSCA', 'LG-FSCA', 'SGO-FSCA', 'SGD-FSCA']
opfs_names = ['OPFS', 'SGO-OPFS', 'SGD-OPFS']
ufs_names = ['UFS', 'LG-UFS', 'SG-UFS']
greedy_names = ['FSCA', 'OPFS', 'UFS']
lazy_names = ['LG-FSCA', 'LG-UFS']
greedy_lazy_names = ['FSCA', 'OPFS', 'UFS', 'LG-FSCA', 'LG-UFS']
stochastic_names = ['SGO-FSCA', 'SGD-FSCA', 'SGO-OPFS', 'SGD-OPFS', 'SG-UFS']

get_names = {'all': algorithm_names, 'fsca': fsca_names, 'opfs': opfs_names, 'ufs': ufs_names, 'g': greedy_names, 'lg': lazy_names, 'glg':greedy_lazy_names,  'sg': stochastic_names}

Nc = 6

datasets = {
    'X50sites': 'data/realData/X50sites.txt',
    'Xpitprops': 'data/realData/Xpitprops.txt',
    'frogs': 'data/realData/frogs.txt',
    'breastCancerDiagnosis': 'data/realData/wdbc.txt',
    'dryBeans': 'data/realData/dryBeans.txt'
}

def random_duration(alg_type):
    names = get_names[alg_type] 
    duration = defaultdict(list)
    compID = defaultdict(list)
    x = []

    for i in range(1,11):
        mat = read_matrix('data/randomData/t{0}.txt'.format(i))
        x.append(mat.shape[1])

        for alg in names:
            res = select_features(alg, mat, Nc)
            duration[alg].append(res.duration)
            compID['{0}_{1}'.format(alg, x)].append(res.component_id)
    
    return x, duration

def real_results(alg_type, ds, percentage=0.5):
    names = get_names[alg_type]
    data_path = datasets[ds]
    mat = read_matrix(data_path)

    duration = defaultdict(list)
    varEx = defaultdict(list)
    compID = defaultdict(list)

    for alg in names:
        res = select_features(alg, mat, Nc, percentage)
        duration[alg].append(res.duration)
        varEx[alg] = res.variance_explained
        compID[alg] = res.component_id

    # test output with print statements 
    for k in duration.keys():
        print(k)
        print(duration[k])
        print(varEx[k])
        print(compID[k])
        print('\n')
    
    return ds, duration, varEx, compID

def real_sg_compare_results(ds, percentages):
    # Get the dataset
    names = get_names['sg']
    data_path = datasets[ds]
    mat = read_matrix(data_path)

    # Setup storage for results
    # duration[alg_type][percentage] = list of durations 
    duration = defaultdict(lambda: defaultdict(list))
    # varEx[alg_type][percentage] = list of explained variances
    varEx = defaultdict(lambda: defaultdict(list))
    # compID[alg_type][percentage] = list of component IDs
    compID = defaultdict(lambda: defaultdict(list))

    # for each stochastic greedy algorithm represented by its name in the names list
    for alg in names:
        
        # select Nc features using p percent of the data each iteration
        for p in percentages:
            res = select_features(alg, mat, Nc, p)
            duration[alg][p] = (res.duration)
            varEx[alg][p] = (res.variance_explained)
            compID[alg][p] = (res.component_id)

    # test output with print statements 
    """ for k in duration.keys():
        print(k)
        print(duration[k])
        print(varEx[k])
        print(compID[k])
        print('\n') """

    return ds, duration, varEx, compID

