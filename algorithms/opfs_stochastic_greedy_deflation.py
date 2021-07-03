import numpy as np 
from helpers.pca import pca_first_nipals

def opfs_stochastic_greedy_deflation(X, Nc=1, percentage=0.5):
    """This function implements the Orthogonal Principal Feature Selection algorithm with
    Stochastic Greedy (also known as lazier than lazy greedy) optimisation applied.
    At each iteration a random sample of the original data is taken and is used to 
    select the next feature. For more than one component, the indices of features selected
    in previous iterations are used to orthogonalise the features in the current subset using
    the Gram-Schmidt process.

    Args:
        X (A 2D numpy array): The matrix m x v -> m is measurements, v is variables
        Nc (Int): The number of components to select
        percentage (int, optional): If random sampling occurs with replacement - this is the percentage
            of data selected from the original data. Defaults to 0.5

    Returns:
        S: The column vectors of each selected feature during each iteration
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features 
    """
    # Matrix needs to be zero mean
    mX = X.mean(axis=0)
    if(max(mX) > 10**-6):
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX

    sample_size = int(X.shape[0] * percentage)

    def get_sample_idxs():
        """Returns a random sample of the row indices in data matrix X

        Returns:
            list: A random sample of the row indices in X
        """
        return np.random.randint(low = 0, high = X.shape[1], size = sample_size, dtype=int)

    # Initialize storage variables
    S = []
    M = []
    VarEx = []
    compID = []
    VEX = 0
    VT = 0
    YhatP = 0

    # Keep track of olumns not yet selected
    col_idxs = np.arange(X.shape[1])

    # Initialise storage for eigen vectors
    EFS = np.zeros(len(col_idxs))

    # Data subset for this iteration
    y_idxs = get_sample_idxs()
    Y = np.take(X, y_idxs, axis=0)

    # First component
    # Column vector containing variance for each column
    VT =  VT + np.var(Y)

    # Calculate scores of 1st pc for curr_Y using nipals algorithm
    t1 = pca_first_nipals(Y[:,col_idxs])

    # Maximise efs
    for i in range(len(col_idxs)):
        x = np.atleast_2d(Y[:,col_idxs[i]]).T
        EFS[i] = np.divide( np.square(np.matmul(x.T, t1)), np.matmul(x.T,x) + np.finfo(float).eps)

    # Select variable most correlated with first pc
    #EFS[compID] = np.nan
    idx = np.nanargmax(EFS)
    x = np.atleast_2d(Y[:,col_idxs[idx]]).T

    # Variance explained using matrix deflation
    th = np.matmul(np.linalg.pinv(x), Y)
    Yhat = np.matmul(x, th)
    YhatP =  YhatP + Yhat

    # Accumulated variance explained
    VEX = VEX + np.divide(np.var(Yhat), VT) * 100

    # Store results
    compID.append(col_idxs[idx])
    VarEx.append(VEX)

    # Update col_idxs bby removing index of selected feature
    col_idxs = np.delete(col_idxs, idx)
    
    # Loop for remaining components
    for _ in range(1, Nc):
        EFS = np.zeros(len(col_idxs))

        # Data subset for this iteration
        y_idxs = get_sample_idxs()
        Y = np.take(X, y_idxs, axis=0)
        
        # Perform deflation step using the already selected columns
        for id in compID:
            x = np.atleast_2d(Y[:,id]).T
            th = np.matmul(np.linalg.pinv(x), Y)
            Yhat = np.matmul(x, th)
            Y = Y - Yhat
            M.append(th)
        
        # Column vector containing variance for each column
        VT =  VT + np.var(Y)

        # Calculate scores of 1st pc for curr_Y using nipals algorithm
        t1 = pca_first_nipals(Y[:,col_idxs])

        # Maximise efs
        for i in range(len(col_idxs)):
            x = np.atleast_2d(Y[:,col_idxs[i]]).T
            EFS[i] = np.divide( np.square(np.matmul(x.T, t1)), np.matmul(x.T,x) + np.finfo(float).eps)

        # Select variable most correlated with first pc
        #EFS[compID] = np.nan
        idx = np.nanargmax(EFS)
        x = np.atleast_2d(Y[:,col_idxs[idx]]).T

        # Variance explained using matrix deflation
        th = np.matmul(np.linalg.pinv(x), Y)
        Yhat = np.matmul(x, th)

        # Accumulated variance explained
        VEX =  VEX + np.divide(np.var(Yhat), VT) * 100

        # Store results
        compID.append(col_idxs[idx])
        VarEx.append(VEX)

        col_idxs = np.delete(col_idxs, idx)
    S = X[:,compID]
    return S, M, VarEx, compID