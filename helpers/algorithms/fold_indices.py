import numpy as np
from sklearn.utils.validation import check_random_state

def split_without_replacement( n_samples, n_splits):
    """ Based on the _iter_test_indices from sklearn class KFold.
        Each split is unique, there are no duplicate values between the splits - no overlap.
        Each index is covered exactly once in the split.

    Args:
        n_samples (Int): The number of samples to be split
        n_splits (Int): The number of splits to make in the samples

    Returns:
        A list of n_split numpy arrays
    """
    # 
    indices = np.arange(n_samples)
    check_random_state(seed=None).shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    fold_idxs = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        fold_idxs.append(np.array(indices[start:stop]))
        current = stop
    return fold_idxs

def split_with_replacement(n_samples, n_split, percentage):
    """ Allows for overlapping values between the splits. 

    Args:
        n_samples (Int): The number of samples to be split
        n_splits (Int): The number of splits to make in the samples
        percentage (Int): The percentage of original data to randomly put in each split.

    Returns:
        A list of n_split numpy arrays
    """
    fold_idxs = []
    idx_range = np.arange(n_samples)

    for i in range(n_split):
        n =int(n_samples * percentage)
        idxs = np.random.randint(low=0, high=n_samples, size=n, dtype=int)
        fold_idxs.append(np.asarray(idxs))
    return fold_idxs