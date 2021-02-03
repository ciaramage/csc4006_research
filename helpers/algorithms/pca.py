from statsmodels.multivariate.pca import PCA

# return first components -> uses svd by default
# does not normalize, standardize, oe demean
def pca_first_components(X): # X = m x v, where m= no samples, v= no variables 
    pca = PCA(X, ncomp=1) 
    return pca.scores, len(pca.cols)

def pca_normalised(X, Nc=1):
    # X = m x v, where m= no samples, v= no variables
    pca = PCA(X, ncomp=Nc, normalize=True) # uses svd by default
    return pca.scores, len(pca.cols)

def pca_nipals(X, Nc=1):
    # X = m x v, where m= no samples, v= no variables
    pca = PCA(X, ncomp=Nc, method='nipals', tol=1e-6)
    
    # return pca scores vector (m x 1)
    return pca.scores, len(pca.cols) #number of columns -> number of iterations? TODO check this out