# Miscellaneous utility functions

import numpy as np
import scipy.special as sc
import cvxpy as cp

def estimate_users(M, V, Q=1):
    """
    Estimates user points with regularized least squares.
    
    Inputs:
        M: d x d metric
        V: d x N pseudo-ideal points
        Q: parameter specifying energy regularization. For each user solves:
            If Q is scalar:
                    argmin_u ||v - (-2M)u||^2 + Q ||u||^2
                = -2 (4 M^T M + Q I)^{-1} * M^T v

            If Q is PSD matrix:
                    argmin_u ||v - (-2M)u||^2 + u^T Q u
                = -2 (4 M^T M + Q)^{-1} M^T v

    Returns: Tikhonov regularized estimate of user point matrix U
    """
    
    d = M.shape[0]
    
    if np.size(Q) == 1:
        Qmat = Q * np.eye(d)
    else:
        Qmat = Q

    return -2 * np.linalg.inv(4 * M.T @ M + Qmat) @ M.T @ V

def unquantized_pairs(X, S, M, V):
    """
    Generates exact difference of distances (i.e., unquantized) paired comparisons from inputs
    
    Inputs:
        X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)
        S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
           tuple containing the first and second pair item indices. e.g., (k, (i,j))
        M: d x d metric
        V: d x N matrix of user points, one column per user

    Returns:
        Y: (len(S),) numpy vector of unquantized measurements
    """

    mT = len(S)
    Y = np.zeros(mT)

    for mi in range(mT):
        k = S[mi][0]
        v = V[:, k]

        i,j = S[mi][1]
        xi = X[:, i]
        xj = X[:, j]
        
        Y[mi] = xi.dot(M.dot(xi)) - xj.dot(M.dot(xj)) + v.dot(xi - xj)

    return Y


def optloss(X, S, Y, M, V, loss_fun, noise_param):
    """
    Generates optimization loss function for noisy one-bit paired comparisons

    Inputs:
        X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)
        S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
           tuple containing the first and second pair item indices. e.g., (k, (i,j))
        Y: # comparisons list of responses (-1 for first item, +1 for second item)
        M: d x d metric (cvxpy Variable or numpy array)
        V: d x N matrix of user points (cvxpy Variable)
        loss_fun: loss function for optimization. From {'hinge', 'logistic'}
        noise_param: noise parameter, if relevant (in 'logistic' loss function, is scaling parameter)

    Returns:
        loss: cvxpy loss function
    """

    if loss_fun != 'logistic' or noise_param is None:
        logistic_scale = 1
    else:
        logistic_scale = noise_param

    # dictionary of loss functions to choose from
    loss_funs = {'hinge': lambda x : cp.pos(1 - x), # cp.pos(x) = max(0, x)
                'logistic': lambda x: cp.logistic(-logistic_scale * x)} # cp.logistic(x) = log(1 + exp(x))

    cp_loss_fun = loss_funs[loss_fun] # load loss function

    mT = len(S)
    loss = 0
    for mi in range(mT):
        k = S[mi][0]

        i,j = S[mi][1]
        xi = X[:, i]
        xj = X[:, j]
        
        loss += (1/mT) * cp_loss_fun(Y[mi] * (cp.quad_form(xi, M) - cp.quad_form(xj, M) + V[:, k] @ (xi - xj)))

    return loss


def one_bit_pairs(X, S, M, V, noise_type='none', noise_param=None):
    """
    Generates one-bit paired comparisons from inputs
    
    Inputs:
        X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)
        S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
           tuple containing the first and second pair item indices. e.g., (k, (i,j))
        M: d x d metric
        V: d x N matrix of user points, one column per user
        noise_type: type of noise in measurement model, from {'none', 'logistic'}
        noise_param: noise parameter, if relevant (in 'logistic' noise, is scaling parameter)

    Returns (tuple):
        Y: one-bit measurements (possibly noisy)
        Y_noiseless: noiseless one-bit measurements
        Y_unquant: unquantized measurements
    """

    Y_unquant = unquantized_pairs(X, S, M, V)
    mT = len(Y_unquant)

    Y_noiseless = np.sign(Y_unquant) # quantize measurements into noiseless one-bit comparisons

    if noise_type == 'none':
        Y = Y_noiseless

    elif noise_type == 'logistic':

        if noise_param is None:
            logistic_scale = 1
        else:
            logistic_scale = noise_param

        pY1 = sc.expit(logistic_scale * Y_unquant)
        u = np.random.rand(mT)
        Y = np.sign(pY1 - u) # P(Y = 1) = P(pY1 - u > 0) = P(u < pY1) = pY1

    # if any 0 entries (almost surely not the case), sets to a coin flip
    Y0mask = Y == 0
    if any(Y0mask):
        Y[Y0mask] = -1 + 2*np.random.randint(2, size=sum(Y0mask))

    return Y, Y_noiseless, Y_unquant


def predict(X, S, M, V):
    """
    Predict one-bit paired comparison results from data

    Inputs:
        X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)
        S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
           tuple containing the first and second pair item indices. e.g., (k, (i,j))
        M: d x d metric
        V: d x N matrix of user points, one column per user

    Returns:
        Ypred: list of predicted one-bit measurements
    """

    Ypred, _, _ = one_bit_pairs(X, S, M, V, noise_type='none')
    return Ypred


def projPSD(M):
    """
    Project symmetric matrix M onto positive semidefinite cone
    """

    Ms = (M + M.T)/2 # make sure Ms is symmetric

    lams, V = np.linalg.eig(Ms)
    lams_trunc = np.maximum(lams, 0)

    return V @ np.diag(lams_trunc) @ V.T