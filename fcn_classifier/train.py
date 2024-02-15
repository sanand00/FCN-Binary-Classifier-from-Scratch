import numpy as np
from functools import partial
from alive_progress import alive_bar
from .utils import propogation, l2_propogation

def train(X, y, W_dict, b_dict, L, alpha, lam, batch_size, max_iter, print_cost = True):
    len_c = max_iter + 1
    C = np.empty(len_c)
    n_samples = X.shape[1]
    n_batches = int(np.ceil(n_samples / batch_size))
    batch_inds = np.append(np.arange(0, n_samples, step = batch_size), n_samples + 1)
    
    if lam == 0:
        propogate = partial(propogation, L = L, alpha = alpha)
    elif lam > 0:
        propogate = partial(l2_propogation, L = L, alpha = alpha, lam = lam)
    else:
        raise ValueError("lam must be a number between 0 and 1, inclusive.")
    
    W = W_dict
    b = b_dict
    if print_cost:
        with alive_bar(len_c, title='Processing', force_tty = True, length = 20, enrich_print = True) as bar:
            for i in range(len_c):
                for j in range(n_batches):
                    lo = batch_inds[j]
                    hi = batch_inds[j+1]
                    W, b, cost = propogate(X[:,lo:hi], y[:, lo:hi], W, b, last_iter = (i == len_c))
                    C[i] += cost 
                C[i] = C[i] / batch_size
                print(f'cost = {C[i]}')
                bar()
    else:
        with alive_bar(len_c, title='Processing', force_tty = True, length = 20) as bar:
            for i in range(len_c):
                for j in range(n_batches):
                    lo = batch_inds[j]
                    hi = batch_inds[j+1]
                    W, b, cost = propogate(X[:,lo:hi], y[:, lo:hi], W, b, last_iter = (i == len_c))
                    C[i] += cost 
                C[i] = C[i] / batch_size
                bar()        
    return W, b, C