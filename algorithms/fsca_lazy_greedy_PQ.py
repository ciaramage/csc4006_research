import numpy as np
from helpers.dataStructures.PriorityQueue import PriorityQueue

def lazy_greedy_FSCA_PQ(X, Nc=1):
    """ This function implements the Forward Selection Component Analysis algorithm with
    optimisation based on Minoux's lazy greedy optimization with a priority queue. 
    Avoids reevaluating features with the least correlated rayleigh quotient.
    This optimization uses a heapq to implement a 'priority queue'

    Args:
        X (A 2D numppy array): The matrix m x v -> m is measurements, v is variables
        Nc (int, optional): The number of components to select. Defaults to 1.

    Returns:
        S: The column vectors of each selected feature during each iteration
        M: Used to deflate the matrix at each iteration
        VarEx: The accumulated variance explained with the inclusion of each selected feature
        compID: The component ID of each of the selected features
    """
    #
    # algorithm required to have zero mean columns
    #
    mX = X.mean(axis=1, keepdims=True)
    if max(mX) > 10**-6:
    # columns not mean centered
        print('\nWarning: Data not zero mean... detrending\n')
        X = X - mX
    #
    # size of matrix (m-measurements, v-variables)
    #
    m,v = X.shape
    L = v # L is the number of variables (columns)
    #
    # sum of matrix diagonal
    #
    Y=X
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
    #idxs = np.argsort(-VEX.flatten('C'))
    #gains = VEX[idxs].flatten('C')
    gains = VEX.flatten('C')
    idxs = np.array(range(0, gains.shape[0]))
    #
    # initialise PriorityQueue of sorted gains with items, priorities -> idxs, gains
    #
    pq = PriorityQueue(idxs, gains)
    #
    # first component -> largest gain and its index in the priority queue
    #
    _, firstCompId = pq.largest_item()

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
    compID.append(firstCompId)

    #
    # loop for the remaining components
    #
    for j in range(1, Nc):
        currentPos = j # track how far through the list we are
        bestGain = 0
        bestGainIdx = 0
        #worst gain of previous iteration
        worstGain, worstGainIdx = pq.smallest_item() 

        while True:
            # if priority queue is empty
            if len(pq.priorityQueue) == 0:
                False

            # calculate variance contribution with inclusion of this variable
            # (currentPos)th largest gain
            _, var_idx = pq.nth_largest_item(currentPos) 
            x = np.atleast_2d(Y[:, var_idx]).T
            r = np.matmul(Y.T, x)
            rQ = np.matmul(r.T, np.divide(r, np.matmul(x.T,x)))
            gain = 100 * np.divide(rQ, TR) # convert rayleigh quotient to variance explained
            # add to the priority queue
            pq.add_item(var_idx, gain)

            # best gain
            if gain > bestGain:
                bestGain = gain
                bestGainIdx = var_idx
            
            # worst gain
            if gain < worstGain:
                worstGain = gain
                worstGainIdx = var_idx

            # if the best exact increment found to date is not greater than the upperbound on the
            # next variable contribution, move to the next variable and evaluate it exactly
            # (currentPosition + 1)th largest item
            # so if 
            nextGain, nextGainIdx = pq.nth_largest_item(currentPos+1)

            if bestGain > nextGain:
                break # best gain found
            else: 
                currentPos = currentPos + 1
                if currentPos == L:
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
        compID.append(bestGainIdx) 
        VarEx.append(bestGain.flatten('C')[0] + VarEx[-1])

    return S, M, VarEx, compID
   

def do_lazy_fsca_pq(X, Nc=1):
    return lazy_greedy_FSCA_PQ(X, Nc)
      