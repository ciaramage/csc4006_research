import numpy as np

def lazy_greedy_FSCA(X, Nc=1):
    """FSCA implemented with Minoux's lazy greedy optimization with a 
    priority queue. Avoids reevaluating features with the least correlated raleigh quotient.
    This optimization uses arrays to implement a 'priority queue'

    Args:
        X ([type]): [description]
        Nc (int, optional): [description]. Defaults to 1.

    Returns:
        [S, M, VarEx, compID]: [description]
    """
    #
    # algorithm required to have zero mean columns
    #
    mX = X.mean(axis=1, keepdims=True)
    if(max(mX) > 10**-6):
        # column do not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    #
    # size of matrix (m - measurements, v - variables)
    #
    m,v = X.shape
    L = v # L is number of variables (columns)
    #
    # sum of matrix diagonal
    #
    Y = X
    TR = np.trace(np.matmul(Y.T, Y))
    #
    # initialise storage variables
    #
    S = [] 
    M = []
    VarEx = []
    compID = []
    #
    # first component
    #
    rQ = np.zeros((L,1))
    for i in range(0,L):
        x = np.atleast_2d(Y[:,i]).T # column i
        #
        # rayleigh quotient for first component
        #
        r = np.matmul(Y.T, x)
        rQ[i] = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
    VEX = 100 * np.divide(rQ, TR) # convert rayleigh quotient to variance explained
    #
    # argsort(-array) return indices the put variance in descending sorted order
    #
    idxs = np.argsort(-VEX.flatten('C'))
    gains = VEX[idxs].flatten('C')
    firstCompId = idxs[0] # first index in idxs is the optimum one
    #
    # deflate matrix
    #
    x = np.atleast_2d(Y[:,firstCompId]).T
    th = np.matmul(np.linalg.pinv(x), Y)
    YhatP = np.matmul(x, th)
    Y = Y - YhatP
    #
    # append first component data to storage variables
    #
    S.append(x)
    M.append(th.T)
    VarEx.append(gains[0])
    compID.append(firstCompId+1)  # component idx reflects matlab indexing from 1
    #
    # loop for the remaining components
    #
    for j in range(1,Nc):
        currentPos = j # track how far through the list 
        bestGain = 0
        bestGainIdx = 0
        worstGain = gains[0]
        worstGainIdx = 0
        while True:
            x = np.atleast_2d(Y[:, idxs[currentPos]]).T # column -> value at current position in idxs
            r = np.matmul(Y.T, x)
            # increment in variance contribution with inclusion of this variable
            rQ = np.matmul(r.T, np.divide(r, np.matmul(x.T, x)))
            gains[currentPos] = 100 * np.divide(rQ, TR) # convert rayleigh quotient to variance explained

            # best gain -> value and location
            if gains[currentPos] > bestGain:
                bestGain = gains[currentPos]
                bestGainIdx = idxs[currentPos]

            # worst gain -> value and location
            if gains[currentPos] < worstGain:
                worstGain = gains[currentPos]
                worstGainIdx = idxs[currentPos]

            # if the best exact increment found to data is not greater than the upper bound on the 
            # next variable contribution, move to the next variable and evaluate it exactly
            if bestGain > gains[currentPos+1]:
                break # best gain found
            else:
                currentPos = currentPos+1
                if currentPos == L-1:
                    break
        #
        # deflate matrix for selected feature
        #
        x = np.atleast_2d(Y[:, bestGainIdx]).T
        th = np.matmul(np.linalg.pinv(x), Y)
        Yhat = np.matmul(x, th)
        Y = Y-Yhat
        #
        # store results for selected feature
        #
        S.append(x)
        M.append(th.T)
        compID.append(bestGainIdx + 1)  # component idx reflects matlab indexing from 1
        VarEx.append(bestGain + VarEx[-1])
        #
        #  resort the list of gains and indexes
        #
        if currentPos < L-1: # last index position is L-1
            while worstGain < gains[currentPos+1]:
                currentPos = currentPos+1
                if currentPos == L-1:
                    break
        newIdxs = np.argsort((-gains[0:currentPos].flatten('C')))
        gains[0:currentPos] = gains[newIdxs]
        idxs[0:currentPos] = idxs[newIdxs]

    return S, M, VarEx, compID

def do_lazy_fsca(X, Nc=1):
    return lazy_greedy_FSCA(X, Nc)