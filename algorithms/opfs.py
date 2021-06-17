import numpy as np 
from helpers.algorithms.pca import pca_first_nipals

def opfs(X, Nc=1):
    """This function implements the baseline Orthogonal Principal Feature Selection algorithm
    with no optimization applied.

    Args:
        X (A 2D numppy array): The matrix m x v -> m is measurements, v is variables
        Nc (int, optional): The number of components to select. Defaults to 1.

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: Used to deflate the matrix at each iteration
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features 
    """
    Y = X.copy()
    # Algorithm requires matrix X to have zero mean columns
    mX = X.mean(axis=0) 
    if( max(mX) > 10**-6):
        # Columns not mean centered
        print("\nWarning: Data not zero mean... detrending\n")
        X = X- mX
        
    # Number of features (columns) in matrix X
    L = X.shape[1] 

    # Sum of column variance
    VT= np.var(Y)

    # Keep track of columns not yet selected
    col_idxs = np.arange(X.shape[1])
    
    # Initialize storage variables
    compID = []
    VarEx = []
    YhatP = 0
    VEX = 0
    M = []
    
    for _ in range(0,Nc):
        EFS=np.zeros(len(col_idxs))
        # Calculate scores of 1st principle component for Y using nipals algorithm
        t1  = pca_first_nipals(Y[:,col_idxs])
        for i in range(len(col_idxs)):
            # Column col_idxs[i]
            x = np.atleast_2d(Y[:,col_idxs[i]]).T

            # Addition of machine float epsilon prevents division by zero
            EFS[i] = np.divide( np.square(np.matmul(x.T, t1)), np.matmul(x.T,x) + np.finfo(float).eps)

        # Maximise the eigenvectors - the variable most correlated with first principal component
        idx = np.nanargmax(EFS) # index of variable with max EFS
        x = np.atleast_2d(Y[:,col_idxs[idx]]).T

        # Deflate matrix
        th = np.matmul(np.linalg.pinv(x),Y)
        Yhat = np.matmul(x, th)
        Y = Y-Yhat
        
        # Calculate accumulated variance explained
        YhatP = YhatP + Yhat
        VEX= np.divide(np.var(YhatP), VT) *100
        
        # Store results
        M.append(th.T)
        compID.append(col_idxs[idx])
        VarEx.append(VEX)

        # Update col_idxs bby removing index of selected feature
        col_idxs = np.delete(col_idxs, idx)

    S = X[:,compID]
    return S, M, VarEx, compID