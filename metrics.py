# Calculates performance metrics for models

import math
import numpy as np

def relative_error(Atrue, Ahat):
    return np.linalg.norm(Atrue - Ahat, 'fro') / np.linalg.norm(Atrue, 'fro')

def scaled_error(Atrue, Ahat):
    return np.linalg.norm(Atrue / np.linalg.norm(Atrue, 'fro') - Ahat / np.linalg.norm(Ahat, 'fro'), 'fro')

def relative_M_error(Mtrue, Mhat):
    return relative_error(Mtrue, Mhat)

def relative_V_error(Mtrue, Utrue, Vhat):
    Vtrue = -2 * Mtrue @ Utrue
    return relative_error(Vtrue, Vhat)

def relative_U_error(Mtrue, Utrue, Uhat):
    Utrue_proj = np.linalg.pinv(Mtrue, rcond=1e-4) @ Mtrue @ Utrue
    return relative_error(Utrue_proj, Uhat)

def scaled_M_error(Mtrue, Mhat):
    return scaled_error(Mtrue, Mhat)

def scaled_V_error(Mtrue, Utrue, Vhat):
    Vtrue = -2 * Mtrue @ Utrue
    return scaled_error(Vtrue, Vhat)

def prediction_accuracy(model, X, S, Y):
    Y_pred = model.predict(X=X, S=S)
    return np.mean(Y_pred == Y)

