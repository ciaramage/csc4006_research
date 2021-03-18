import numpy as np
from queue import PriorityQueue

def lazy_greedy_FSCA(X, Nc=1):
    """FSCA implemented with Minoux's lazy greedy optimization with a 
    priority queue

    Args:
        X ([type]): [description]
        Nc (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    S = []
    M = []
    VarEx = []
    compID = []
    return S, M, VarEx, compID