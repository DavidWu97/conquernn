import numpy as np


def create_folds(X, k):
    if isinstance(X, int) or isinstance(X, np.integer):
        indices = np.arange(X)
    elif hasattr(X, '__len__'):
        indices = np.arange(len(X))
    else:
        indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in range(k):
        start = end
        end = start + len(indices) // k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def fold_complement(N, fold):
    '''Returns all the indices from 0 to N that are not in fold.'''
    mask = np.ones(N, dtype=bool)
    mask[fold] = False
    return np.arange(N)[mask]

def train_test_split(test_indices, *splittables):
    if len(splittables) == 0:
        return []
    N = len(splittables[0])
    train_indices = fold_complement(N, test_indices)
    return ([v[train_indices] for v in splittables],
            [v[test_indices] for v in splittables])

def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    nbatches = int(np.ceil(len(order) / float(batch_size)))
    for b in range(nbatches):
        idx = order[b*batch_size:min((b+1)*batch_size, len(order))]
        yield idx

def get_idx(sceidx,shapeidx,nidx):
    h_index = [[[0, 0, 0],[3, 1, 1]],
               [[2, 1, 0],[4, 1, 0]],
               [[1, 1, 0],[2, 0, 0]]]
    return h_index[sceidx][shapeidx][nidx]

def filter_h(data):

    # data shape:   (trial, scenario, model shape, bandwidth, kernel, sample size, quantile)
    # or            (scenario, model shape, bandwidth, kernel, sample size, quantile, 100)
    # filter h by scenario, model shape and sample size
    
    if data.shape[0] == 50:
        out = np.zeros((50,3,2,4,3,5))
        for sceidx in range(3):
            for shapeidx in range(2):
                for nidx in range(3):
                    hidx = get_idx(sceidx,shapeidx,nidx)
                    out[:,sceidx,shapeidx,:,nidx,:] = data[:,sceidx,shapeidx,hidx,:,nidx,:]
    else:
        out = np.zeros((3,2,4,3,5,100))
        for sceidx in range(3):
            for shapeidx in range(2):
                for nidx in range(3):
                    hidx = get_idx(sceidx,shapeidx,nidx)
                    out[sceidx,shapeidx,:,nidx,:,:] = data[sceidx,shapeidx,hidx,:,nidx,:,:]
    return out