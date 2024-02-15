import numpy as np

def ReLU(x):
    return np.maximum(x, 0)
    
def sigmoid(x):
    return 1 / (1+ np.exp(-x))

def sigma(x, L, layer):
    if layer == L:
        return sigmoid(x)
    else:
        return ReLU(x)
        
def d_sigma(x):
    return (x > 0).astype(int)
    
def loss(y, yhat):
    loss_vec = np.multiply(y-1, np.log(1 - yhat)) -  np.multiply(y,np.log(yhat))
    return np.mean(loss_vec)


def propogation(X, y, W_dict, b_dict, L, alpha, last_iter = False):
    # Forward propogation
    m = X.shape[1]
    z = {}
    a = {0: X}
    for i in range(1, L+1):
        z[i] = W_dict[i] @ a[i-1] + b_dict[i]
        a[i] = sigma(z[i], L, i)
    C = loss(y, a[L])
    if last_iter:
        return (W_dict, b_dict, C)
    
    # Backward propogation
    delta_curr = (a[L] - y)
    delta_next = 0
    for i in range(L, 0, -1):
        if i != 1:
            delta_next = np.multiply(np.transpose(W_dict[i]) @ delta_curr, d_sigma(z[i-1]))
        W_dict[i] -= alpha / m * delta_curr @ np.transpose(a[i-1]) 
        b_dict[i] -= alpha * np.mean(delta_curr, axis = 1, keepdims = True)
        delta_curr = delta_next
    return (W_dict, b_dict, C)

def l2_propogation(X, y, W_dict, b_dict, L, alpha, lam, last_iter = False):
    # Forward propogation
    m = X.shape[1]
    z = {}
    a = {0: X}
    for i in range(1, L+1):
        z[i] = W_dict[i] @ a[i-1] + b_dict[i]
        a[i] = sigma(z[i], L, i)
    C = loss(y, a[L])
    if last_iter:
        return (W_dict, b_dict, C)  
    
    # Backward propogation
    delta_curr = (a[L] - y)
    delta_next = 0
    for i in range(L, 0, -1):
        if i != 1:
            delta_next = np.multiply(np.transpose(W_dict[i]) @ delta_curr, d_sigma(z[i-1]))
        W_dict[i] -= alpha / m * (delta_curr @ np.transpose(a[i-1])  + lam * W_dict[i])
        b_dict[i] -= alpha * np.mean(delta_curr, axis = 1, keepdims = True)
        delta_curr = delta_next
    return (W_dict, b_dict, C)