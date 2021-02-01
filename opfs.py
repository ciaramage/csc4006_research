import numpy as np 
from sklearn import preprocessing
from numpy.linalg import eigh
import pca_analysis.pca as pca

# reshapes matrix from 
# X.shape() = (nRows, nCols) -> X.shape = (nCols, nRows)
def mat_dash(X):
    (nRows, nCols) = X.shape
    X = np.reshape(X, (nCols, nRows))
    return X

def opfs(X, Nc=1):
    # matrix x needs to have zero mean columns - they should by default in matrix_generator but check anyway
    mX = X.mean(axis = 0) # calculate mean of X across axis=0 (0 for columns, 1 for rows)
    if( max(mX) > 10**-6):
        # columns do not have zero mean then fix
        print("\nWarning: Data not zero mean... detrending\n")
        X = X- np.ones(X.shape)*mX 
    
    #
    # size of matrix (m - measurements, v - variables)
    #
    (m,v) = X.shape 
    L = v # l is number of variables (columns)
    VT=np.var(Y, axis=0) # Y is row vector containing variance for each column

    #
    # initialize storage variables
    #
    compId[]
    VarEx[]
    YhatP = 0
    S = []
    M = []
    #
    # initialise storage for correlation vector
    #
    EFS=np.zeros((L,1))  # eigen factors

    for j in range(1,Nc):
        
        # calculate scores of 1st pc for Y using nipals algorithm
        t1, num_iterations = pca.pca_nipals(Y)

        for i in range(1,L):
            x = Y[:,i] # column i
            x_dash = mat_dash(x)
            EFS(i) = ((x_dash*t1)**2) / ((x_dash*x) + np.finfo(eps))

        #
        # select index of max efs
        #
        idx = np.argmax(EFS)
        x = Y[:,idx]

        #
        # deflate matrix
        #
        th = np.matmul(np.linalg/pinv(x), Y)
        Yhat = np.matmul(x,th)
        Y = Y-Yhat

        # 
        # store results
        #
        S.append(x)
        M.append(mat_dash(th))
        YhatP = YhatP + Yhat
        compId.append(idx)
        VarEx.append(np.var(np.array(YhatP)) / VT*100)
    
    # return S, M, VarEx, compId
    return S, M, VarEx, compID

            

            
