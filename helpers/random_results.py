from algorithms.fsca import FSCA
import numpy as np
import time
from helpers.matrix_in_out import *
from algorithms.fsca import FSCA



def get_random_duration():
    sizes = []
    Nc = 6
    results=[]
    for i in range(1,10):
        mat = read_matrix('data/randomData/t{0}.txt'.format(i+1))
        sizes.append(mat.shape)
        start = time.time()
        _, _, varEx_fsca, compID_fsca = FSCA(mat, Nc)
        duration = time.time() - start
        results.append((mat.shape, duration))


    for result in results:
        print(result)

