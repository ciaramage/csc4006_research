import numpy as np 
import pandas as pd 
from generators.matrix_generator import get_matrix
from algorithms.fsca import do_fsca

square_mat = get_matrix(True)

#for i in range(len(square_mat[0])):
    
S, M, VarEx, compId = FSCA_analysis(square_mat[0])


print(S)

