import numpy as np 
from helpers.pca import pca_first_nipals
from numpy.linalg import norm

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
    varEx = []
    compID = []
    vex = 0
    VT = 0

    # Data subset for this iteration
    y_idxs = get_sample_idxs()
    Y = np.take(X, y_idxs, axis=0)
    
    # Initialise storage for eigen vectors
    EFS = np.zeros(Y.shape[1])

    ## First component
    # Column vector containing variance for each column
    VT = np.var(Y)

    # Calculate scores of 1st pc for curr_Y using nipals algorithm
    pc = pca_first_nipals(Y)

    # Maximise efs
    for i in range(Y.shape[1]):
        # feature column f in Y
        f = np.atleast_2d(Y[:,i]).T
        corr = np.divide( np.matmul(f.T, pc), norm(f))
        EFS[i] = corr

    # Select variable most correlated with first pc
    idx = np.nanargmax(EFS)
    x = np.atleast_2d(Y[:,idx]).T

    # Variance explained using matrix deflation
    th = np.matmul(np.linalg.pinv(x), Y)
    Yhat = np.matmul(x, th)

    # Accumulated variance explained
    vex = np.divide(np.var(Yhat), VT) * 100

    # Store results
    compID.append(idx)
    varEx.append(vex)
    
    # Loop for remaining components
    for _ in range(1, Nc):
        EFS = np.zeros(Y.shape[1])

        # Data subset for this iteration
        y_idxs = get_sample_idxs()
        Y = np.take(X, y_idxs, axis=0)
        
        # Deflation Step
        for id in compID:
            x = np.atleast_2d(Y[:,id]).T
            th = np.matmul(np.linalg.pinv(x), Y)
            Yhat = np.matmul(x, th)
            Y = Y - Yhat
            M.append(th)

        # Calculate scores of 1st pc for remaining columns using nipals algorithm
        pc = pca_first_nipals(Y)

        # Maximise efs
        for i in range(Y.shape[1]):
                # feature column f in Y
            f = np.atleast_2d(Y[:,i]).T
            corr = np.divide( np.matmul(f.T, pc), norm(f))
            EFS[i] = corr

        # Select variable most correlated with first pc
        idx = np.nanargmax(EFS)
        x = np.atleast_2d(Y[:,idx]).T
        
        # Variance explained using matrix deflation
        th = np.matmul(np.linalg.pinv(x), Y)
        Yhat = np.matmul(x, th)

        # Accumulated variance explained
        vex = vex + (np.divide(np.var(Yhat), VT) * 100)

        # Store results
        compID.append(idx)
        varEx.append(vex)

    S = X[:,compID]
    return S, M, varEx, compID