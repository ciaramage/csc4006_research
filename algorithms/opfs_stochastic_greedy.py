import numpy as np 
from sklearn import preprocessing
from numpy.linalg import eigh
from helpers.algorithms.pca import pca_first_nipals

def STOCHASTIC_OPFS(X, Nc=1, replacement=True, percentage=0):

    compID = []
    VarEx = []
    S = []
    M = []

    return S, M, VarEx, compID