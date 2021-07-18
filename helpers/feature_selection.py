import time
# import fsca implementationa
from algorithms.fsca import fsca
from algorithms.fsca_lazy_greedy import fsca_lazy_greedy
from algorithms.fsca_stochastic_deflation import fsca_stochastic_greedy_deflation
from algorithms.fsca_stochastic_greedy import fsca_stochastic_greedy_orthogonal
# import opfs implementations
from algorithms.opfs import opfs
from algorithms.opfs_stochastic_greedy_deflation import opfs_stochastic_greedy_deflation
from algorithms.opfs_stochastic_greedy import opfs_stochastic_greedy_orthogonal
# import ufs implementations
from algorithms.ufs import ufs
from algorithms.ufs_lazy_greedy import ufs_lazy_greedy
from algorithms.ufs_stochastic_greedy import ufs_stochastic_greedy

class Result:
    def __init__(self, name, shape, number_components, variance_explained, component_id, duration, percentage=0.5):
        """ __init__([args]) ensure proper initialisation of the Result class.

        Args:
            name (String): name of algorithm used
            shape (List): shape of data matrix
            number_components (Integer): number of features to select
            variance_explained (List): accumulated variance explained by each component selected
            component_id (List): Column indexes of each selected component
            duration (float): Time taken to execute algorithm
            percentage (float, optional): Percentage of data randomly sampled at each iteration when stochastic greedy optimisation has been applied. Defaults to 0.5.
        """
        self.name = name
        self.shape = shape
        self.number_components = number_components
        self.variance_explained = variance_explained
        self.component_id = component_id
        self.duration = duration
        self.percentage = percentage

def select_features(algorithm_name, X, N, percentage=0.5):
    """ select_feature function will select N features from input data matrix X using the algorithm described by algorithm_name
    The percentage argument will only be used when the algorithm employ stochastic greedy optimisation.

    Args:
        algorithm_name (String): The name of the algorithm to use to select N features from input data matrix X
        X (2D numpy array): Data input matrix
        N (Integer): The number of features to select
        percentage (float, optional): The percentage of data to random sample from the data input matrix X when stochastic greedy optimisation has been applied. Defaults to 0.5.

    Raises:
        ValueError: If the value of the algorithm_name parameter is not any of those used to describe a feature selection algorithm. 
            Acceptable values are [ fsca, fscaLG, fscaSGO, fscaSGD, opfs, opfsSGO, opfsSGD, ufs, ufsLG, ufsSG ]

    Returns:
        res (Result): An instance of the class Result.
    """
    if algorithm_name == 'FSCA': # Greedy Forward Selection Component Analysis
        s = time.perf_counter()
        _,_, v, c = fsca(X,N)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)))
        return res
    elif algorithm_name == 'LG-FSCA': # Lazy Greedy Forward Selection Component Analysis
        s = time.perf_counter()
        _,_, v, c = fsca_lazy_greedy(X,N)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)))
        return res
    elif algorithm_name =='SGO-FSCA': # Stochastic Greedy (with Orthogonalisation) Forward Selection Component Analysis
        s = time.perf_counter()
        _,_, v, c = fsca_stochastic_greedy_orthogonal(X,N, percentage)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)), percentage)
        return res
    elif algorithm_name == 'SGD-FSCA': # Stochastic Greedy ( with Deflation) Forward Selection Component Analysis
        s = time.perf_counter()
        _,_, v, c = fsca_stochastic_greedy_deflation(X,N, percentage)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)), percentage)
        return res
    elif algorithm_name == 'OPFS': # Greedy Orthogonal Principal Feature Selection
        s = time.perf_counter()
        _,_, v, c = opfs(X,N)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)))
        return res
    elif algorithm_name == 'SGO-OPFS': # Stochastic Greedy (with Orthogonalisation) Orthogonal Principal Feature Selection
        s = time.perf_counter()
        _,_, v, c = opfs_stochastic_greedy_orthogonal(X,N, percentage)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)), percentage)
        return res
    elif algorithm_name == 'SGD-OPFS': # Stochastic Greedy (with Deflation) Orthogonal Principal Feature Selection
        s = time.perf_counter()
        _,_, v, c = opfs_stochastic_greedy_deflation(X,N, percentage)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)), percentage)
        return res
    elif algorithm_name == 'UFS': # Greedy Unsupervised Feature Selection
        s = time.perf_counter()
        _,_, v, c = ufs(X,N)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)))
        return res
    elif algorithm_name == 'LG-UFS': # Lazy Greedy Unsupervised Feature Selection
        s = time.perf_counter()
        _,_, v, c = ufs_lazy_greedy(X,N)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)))
        return res
    elif algorithm_name == 'SG-UFS': # Stochastic Greedy Unsupervosed Feature Selection
        s = time.perf_counter()
        _,_, v, c = ufs_stochastic_greedy(X,N, percentage)
        d = time.perf_counter() - s
        res = Result(algorithm_name, X.shape, N, v, c, float("{0:.6f}".format(d)), percentage)
        return res
    else:
        raise ValueError('invalid value given for algorithm_name')

        
