"""
Modeling object for simultaneous metric and preference learning over multiple users
"""

import numpy as np
import cvxpy as cp
import utils
from dataset import Dataset
import sklearn.model_selection as ms
import logging

class MultiPref:
    """
    Modeling object for simultaneous metric and preference learning over multiple users

    Methods:
        fit: fit model with input data
        predict: predict one-bit paired comparisons from data
        get_params: returns model parameters (i.e., M, V, hyperparams) in a dictionary
        estimate_users: estimate user ideal points, after model fitting
        getEmpiricalRisk(self): returns empirical risk of solved model

    Utilities:
        _multipref_wrapper: wrapper function to fit specified constraints
        _multipref_base: base function for multiuser preference learning, accepting constraint
            function as input

        Constraint sets:
            _frobM_l2v_constraints: ||M||F <= lambda0, ||vk||2 <= lambda1 for all k
            _nucM_l2v_constraints: ||M||* <= lambda0, ||vk||2 <= lambda1 for all k
            _nucM_nucV_constraints: ||M||* <= lambda0, ||V||* <= lambda1
            _nucfull_constraints: ||[M, V]||* <= lambda0
            _null: no constraints (beyond M being PSD)
    """
    
    def __init__(self, d, N, method):
        """
        Inputs:
            d: ambient dimension
            N: number of users
            method: modeling algorithm, from {'nucfull', 'nucM_l2v', 'nucM_nucV', 'frobM_l2v'}

        Stored parameters:
            M: d x d metric (initialized as None)
            V: d x N matrix of pseudo-user points (initialized as None), one column per user
        """

        # Dictionary of model methods and associated fit functions.
        #    key: model name
        #    value: dictionary
        #        'fun': utility fit function
        #        'nhyp': number of hyperparameters
        
        fit_methods = {'nucfull': {'fun':self._multipref_wrapper(self._nucfull_constraints), 'nhyp':1},
                       'nucM_l2v': {'fun':self._multipref_wrapper(self._nucM_l2v_constraints), 'nhyp':2},
                       'nucM_nucV': {'fun':self._multipref_wrapper(self._nucM_nucV_constraints), 'nhyp':2},
                       'frobM_l2v': {'fun':self._multipref_wrapper(self._frobM_l2v_constraints), 'nhyp':2},
                       'psdM': {'fun':self._multipref_wrapper(self._null), 'nhyp':0}}

        self.d = d
        self.N = N
        self.M = None
        self.V = None
        self.lams = None # list of hyperparameters, initialized as none
        self.fit_method = fit_methods[method]['fun']
        self.nhyp = fit_methods[method]['nhyp']
        self.valacc = None # validation accuracy
        self.empirical_risk = None

        # saved for debugging purposes
        self.worst_lams = None # worst set of hyperparameters found during validation
        self.worst_valacc = None # worst validation accuracy (achieved by worst_lams)
        
    def fit(self, X, S, Y, lams=None, n_splits=5, hypergrid_type='range',
            hypergrid=np.logspace(-5, 5, 11), blocked=True, verbose=False,
            loss_fun='hinge', sample_size=None, Mfixed=None, noise_param=None,
            cvxkwargs=None, use_logger=False, projPSD=False):
        """
        High-level fit function, calls algorithm-specific fit function

        Inputs:
            X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)

            S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
               tuple containing the first and second pair item indices. e.g., (k, (i,j))

            Y: # comparisons list of responses (-1 for first item, +1 for second item)

            lams: hyperparameters to use for model fitting. If None, will perform cross-validation

            n_splits: number of cross-validation splits

            hypergrid_type: type of hyperparameter search grid (if using cross-validation), from 'range' or 'grid'

            hypergrid: specified hypergrid variable. If hypergrid_type == 'range', specify grid range for each
                       hyperparameter. If hypergrid_type == 'grid', specify exact grid for hyperparameter search

            blocked: flag to group within users during cross-validation

            verbose: flag for printing status updates

            loss_fun: loss function for optimization, from {'hinge', 'logistic'}

            sample_size (float or int): rate or absolute size for downsampling training set
                before cross-validation. If float between 0.0 and 1.0, specifies downsampling rate of training set.
                If int, specifies exact number of training examples. None for no downsampling

            Mfixed: if specified, fixes metric M and only solves for pseudo-ideal points V.
                If None, solves for M as well

            noise_param: noise parameter, if relevant (in 'logistic' loss function, is scaling parameter)

            cvxkwargs: if not None, kwargs for cvxpy solve()

            use_logger: flag to print to logger instead of console

            projPSD: flag indicating whether or not to project final answer (after optimization by CVX)
                onto positive semidefinite cone
        """

        # check dimensionality and number of users
        assert X.shape[0] == self.d
        assert max([s[0] for s in S]) <= self.N - 1

        if use_logger:
            printfun = lambda s : logging.info(s)
        else:
            printfun = lambda s: print(s)

        if sample_size is None:
            S_all = S
            Y_all = Y

        else:
            if blocked:
                stratify = [s[0] for s in S] # stratify by user indices
            else:
                stratify = None

            S_all, _, Y_all, _ = ms.train_test_split(S, Y,
                    train_size=sample_size, shuffle=True, stratify=stratify)

        if lams is None: # performing cross-validation to select hyperparameters

            # make hyperparameter grid
            nhyp = self.nhyp

            if hypergrid_type == 'range':

                lamrange = np.array(hypergrid)
                assert len(lamrange.shape) == 1

                gridres = len(lamrange)
                lamgrid = np.zeros((gridres**nhyp, nhyp)) # hyperparameter grid has nhyp hyperparameters (# columns) and
                #    gridres^nhyp rows, for every combination of hyperparameters

                for li in range(nhyp):
                    lamgrid[:, li] = np.tile(np.repeat(lamrange, gridres**li), gridres**(nhyp-(li+1)))
                    # rationale: column 0 cycles through lamrange exactly, for gridres**nhyp / gridres**1 times
                    #            column 1 cycles through lamrange, but stays fixed for changes in column 0 (so each entry repeats
                    #                gridres times. This cycling repeats gridres**nhyp / gridres**2 times
                    #            ...
                    
            elif hypergrid_type == 'grid':
                lamgrid = np.array(hypergrid)
                assert lamgrid.shape[1] == nhyp

            nsearch = len(lamgrid) # number of hyperparameter combinations to search over

            # cross validation initialization
            lams_best = None # best hyperparameters
            score_best = None # best hyperparameter score (i.e., validation accuracy)
            lams_worst = None # worst hyperparameters (saved for debugging purposes)
            score_worst = None # worst hyperparmateres score (i.e., worst validation accuracy)
            M_best = None # best learned metric
            V_best = None # best learned psuedo-ideal points

            user_ids = [s[0] for s in S_all] # for KFold splitting

            for ni in range(nsearch):

                if verbose:
                    printfun('Evaluating hyperparameter combination {} / {}'.format(ni + 1, nsearch))

                lams_ni = lamgrid[ni, :]
                
                if blocked:
                    kf = ms.StratifiedKFold(n_splits=n_splits, shuffle=True)

                else:
                    kf = ms.KFold(n_splits=n_splits, shuffle=True)

                score = 0
                for train_idx, val_idx in kf.split(X=np.zeros(len(user_ids)), y=user_ids):

                    Strain = [S_all[idx] for idx in train_idx]
                    Ytrain = Y_all[train_idx]
                    Sval = [S_all[idx] for idx in val_idx]
                    Yval = Y_all[val_idx]

                    M, V, _ = self.fit_method(X, Strain, Ytrain, lams_ni,
                        verbose=verbose, loss_fun=loss_fun, Mfixed=Mfixed,
                        noise_param=noise_param, cvxkwargs=cvxkwargs,
                        use_logger=use_logger, projPSD=projPSD)
                    Ypred = utils.predict(X, Sval, M, V)
                    score += (1/n_splits)*np.mean(Ypred == Yval) # average validation accuracy over each split
                
                if score_best is None or score > score_best:
                    score_best = score
                    lams_best = lams_ni
                    M_best = M
                    V_best = V

                if score_worst is None or score < score_worst:
                    score_worst = score
                    lams_worst = lams_ni
            
            self.lams, self.valacc , self.worst_lams, self.worst_valacc = (
                lams_best, score_best, lams_worst, score_worst) # save validation data

        else: # lams specified, no need for cross-validation
            assert len(lams) == self.nhyp # sanity check on number of hyperparameters
            self.lams = lams

        # fit on all training data with selected hyperparameters
        self.M, self.V, self.empirical_risk = self.fit_method(X, S_all, Y_all, self.lams,
            verbose=verbose, loss_fun=loss_fun, Mfixed=Mfixed,
            noise_param=noise_param, cvxkwargs=cvxkwargs,
            use_logger=use_logger, projPSD=projPSD)

    def predict(self, X, S):
        """
        Predict one-bit paired comparisons from data
        
        Inputs:
            X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)
            S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
               tuple containing the first and second pair item indices. e.g., (k, (i,j))

        Returns: # comparisons list of predicted responses (-1 for first item, +1 for second item)
        """

        return utils.predict(X, S, self.M, self.V)

    def get_params(self):
        """
        Returns model parameters in dictionary with keys ['M', 'V', 'hyperparams']
        Here, 'hyperparams' is the model hyperparameters (initialized as None)
        """

        return {'M':self.M, 'V': self.V, 'hyperparams':self.lams}
    
    def estimate_users(self, Q=1):
        """
        Estimates user points with regularized least squares.
        
        Inputs:
            Q: parameter specifying energy regularization. For each user solves:
                If Q is scalar:
                      argmin_u ||v - (-2M)u||^2 + Q ||u||^2
                    = -2 (4 M^T M + Q I)^{-1} * M^T v

                If Q is PSD matrix:
                      argmin_u ||v - (-2M)u||^2 + u^T Q u
                    = -2 (4 M^T M + Q)^{-1} M^T v

        Returns: Tikhonov regularized estimate of user point matrix U
        """

        return utils.estimate_users(M=self.M, V=self.V, Q=Q)

    def getEmpiricalRisk(self):
        # Returns empirical risk of solved model
        return self.empirical_risk

    def _multipref_wrapper(self, confun):
        """
        Wrapper function to make multipref fit method

        Inputs:
            confun: constraint generation function (e.g., self._frobM_l2v_constraints)

        Returns: fit function ready for use in fit_methods dict (see initialization)
        """

        return (lambda X, S, Y, lams, verbose, loss_fun, Mfixed, noise_param,
                cvxkwargs, use_logger, projPSD:
                self._multipref_base(X, S, Y, confun, lams, verbose, loss_fun,
                Mfixed, noise_param, cvxkwargs, use_logger, projPSD))

    def _multipref_base(self, X, S, Y, confun, lams, verbose=False,
        loss_fun='hinge', Mfixed=None, noise_param=None, cvxkwargs=None,
        use_logger=False, projPSD=False):
        """
        Base function for multiuser preference learning, accepting constraint
        function as input.

        Inputs:

            X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)

            S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
               tuple containing the first and second pair item indices. e.g., (k, (i,j))

            Y: # comparisons list of responses (-1 for first item, +1 for second item)

            confun: constraint generation function (e.g., self._frobM_l2v_constraints)

            lams: hyperparameters to use for model fitting

            verbose: flag for printing status updates

            loss_fun: loss function for optimization, from {'hinge', 'logistic'}

            Mfixed: if specified, fixes metric M and only solves for pseudo-ideal points V.
                If None, solves for M as well. WARNING: does not check PSD or optimization program constraints on input

            noise_param: noise parameter, if relevant (in 'logistic' loss function, is scaling parameter)

            cvxkwargs: if not None, kwargs for cvxpy solve()

            use_logger: flag to print to logger instead of console

            projPSD: flag indicating whether or not to project final answer (after optimization by CVX)
                onto positive semidefinite cone

        Returns: learned parameters in tuple, i.e., (M, V), where:
            M: d x d metric
            V: d x N matrix of pseudo-user points, one column per user
        """

        if use_logger:
            printfun = lambda s : logging.info(s)
        else:
            printfun = lambda s: print(s)

        d = self.d
        N = self.N

        if Mfixed is None:
            M = cp.Variable((d, d), PSD=True) # M is symmetric and positive semidefinite
        else:
            M = Mfixed.copy()

        V = cp.Variable((d, N))

        if verbose:
            printfun('    Constructing loss...')
    
        loss = utils.optloss(X, S, Y, M, V, loss_fun, noise_param)

        if verbose:
            printfun('    Adding constraints...')

        constraints = confun(M, V, lams)

        if verbose:
            printfun('    Setting up problem...')

        prob = cp.Problem(cp.Minimize(loss), constraints)

        if verbose:
            printfun('    Solving...')

        if cvxkwargs is None:
            prob.solve()
        else:
            prob.solve(**cvxkwargs)

        if verbose:
            if prob.status == 'optimal':
                printfun('    Solved!')
            else:
                printfun('    Solved, but not optimally.')
            printfun('')

        if Mfixed is None:
            M = M.value

            if projPSD:
                M = utils.projPSD(M)

            M_eigs = np.linalg.eig(M)[0]
            M_max_abs_eigs = max(np.abs(M_eigs))
            M_min_eig = min(M_eigs)
            tol = -1e-6
            ratio = M_min_eig / M_max_abs_eigs

            if ratio < tol:
                printfun(
                'Min eigenvalue is {:.2e}, '.format(M_min_eig) + \
                'max eigenvalue magnitude is {:.2e}, '.format(M_max_abs_eigs) + \
                'ratio is {:.2e}, crossed below threshold of {:.2e}'.format(ratio ,tol)
                )

        return M, V.value, prob.value

    def _frobM_l2v_constraints(self, M, V, lams):
        """
        Returns constraints for frobM_l2v method:
            ||M||F <= lambda0, ||vk||2 <= lambda1 for all k

        Inputs:
            M: d x d metric (cvxpy Variable)
            V: d x N matrix of user points (cvxpy Variable)
            lams: hyperparameters to use for model fitting

        Returns: constraint list for cvxpy
        """

        constraints = [cp.norm(M, 'fro') <= lams[0]]
        N = V.shape[1]

        for k in range(N):
            constraints.append(cp.norm(V[:, k], 2) <= lams[1])
        
        if type(M) == type(V): # proxy for M being cvxpy variable
            constraints.append(M >> 0)

        return constraints
    
    def _nucM_l2v_constraints(self, M, V, lams):
        """
        Returns constraints for nucM_l2v method:
            ||M||* <= lambda0, ||vk||2 <= lambda1 for all k

        Inputs:
            M: d x d metric (cvxpy Variable)
            V: d x N matrix of user points (cvxpy Variable)
            lams: hyperparameters to use for model fitting

        Returns: constraint list for cvxpy
        """

        constraints = [cp.norm(M, 'nuc') <= lams[0]]
        N = V.shape[1]

        for k in range(N):
            constraints.append(cp.norm(V[:, k], 2) <= lams[1])
        
        if type(M) == type(V): # proxy for M being cvxpy variable
            constraints.append(M >> 0)

        return constraints
        
    def _nucM_nucV_constraints(self, M, V, lams):
        """
        Returns constraints for nucM_nucV method:
            ||M||* <= lambda0, ||V||* <= lambda1

        Inputs:
            M: d x d metric (cvxpy Variable)
            V: d x N matrix of user points (cvxpy Variable)
            lams: hyperparameters to use for model fitting

        Returns: constraint list for cvxpy
        """

        constraints = [cp.norm(M, 'nuc') <= lams[0], cp.norm(V, 'nuc') <= lams[1]]

        if type(M) == type(V): # proxy for M being cvxpy variable
            constraints.append(M >> 0)

        return constraints

    def _nucfull_constraints(self, M, V, lams):
        """
        Returns constraints for nucfull method:
            ||[M, V]||* <= lambda0

        Inputs:
            M: d x d metric (cvxpy Variable)
            V: d x N matrix of user points (cvxpy Variable)
            lams: hyperparameters to use for model fitting

        Returns: constraint list for cvxpy
        """

        constraints = [cp.norm(cp.hstack([M, V]), 'nuc') <= lams[0]]

        if type(M) == type(V): # proxy for M being cvxpy variable
            constraints.append(M >> 0)

        return constraints

    def _null(self, M, V, lams):
        """
        Returns empty constraint set
        """

        constraints = []

        if type(M) == type(V): # proxy for M being cvxpy variable
            constraints.append(M >> 0)

        return constraints


if __name__ == '__main__':

    ### Example usage on normally distributed data

    np.random.seed(10)

    #print('Normally distributed data')
    d,r,N = 3,1,3
    lam = np.sqrt(r*(d**2 + d*N))

    normal_dataset = Dataset('Normal', noise_type='logistic', noise_param=10, d=d, r=r, m=100, N=N, n=100)
    data = normal_dataset.getAllData()
    X, S_all, Y_all, Mtrue, Utrue = data['X'], data['S'], data['Y'], data['M'], data['U']
    S_train, S_test, Y_train, Y_test = normal_dataset.getTrainTestSplit()

    Vtrue = -2 * Mtrue @ Utrue

    # initialize model object
    normal_model = MultiPref(d=normal_dataset.pm['d'], N=normal_dataset.pm['N'], method='nucfull')
    normal_model.fit(X, S_train, Y_train, lams=[lam], verbose=True,
        loss_fun='hinge', cvxkwargs={'solver':cp.SCS, 'eps':1e-6}, projPSD=True)
    Uhat = normal_model.estimate_users()

    # predict training and test labels
    Y_train_pred = normal_model.predict(X, S_train)
    Y_test_pred = normal_model.predict(X, S_test)

    print('Data noise: {:.2%}'.format(np.mean(normal_dataset.Y_noiseless != Y_all)))
    print('Training accuracy, n={}: {:.2%}'.format(len(Y_train), np.mean(Y_train_pred == Y_train)))

    if normal_model.valacc is not None:
        print('Validation accuracy: {:.2%}'.format(normal_model.valacc))

    print('Testing accuracy, n={}: {:.2%}'.format(len(Y_test), np.mean(Y_test_pred == Y_test)))
    print()

    ### Another example with color data
    print('Color data')

    colordata = Dataset('Color', N=5, color_path='./CPdata.mat') # 5 users
    data = colordata.getAllData()
    X, S_all, Y_all = data['X'], data['S'], data['Y']
    S_train, S_test, Y_train, Y_test = colordata.getTrainTestSplit(train_size=500)

    max_x_norm = max(np.linalg.norm(X, axis=0))
    lams_frobM_l2v = [np.sqrt(3), 2* max_x_norm]

    # fit model
    color_model = MultiPref(d=colordata.pm['d'], N=colordata.pm['N'], method='frobM_l2v')
    color_model.fit(X, S_train, Y_train, verbose=True, lams=lams_frobM_l2v,
        loss_fun='hinge', n_splits=3, cvxkwargs={'solver':cp.CVXOPT}, projPSD=True)

    # predict training and test labels
    Y_train_pred = color_model.predict(X, S_train)
    Y_test_pred = color_model.predict(X, S_test)

    print('Training accuracy, n={}: {:.2%}'.format(len(Y_train), np.mean(Y_train_pred == Y_train)))
    print('Testing accuracy, n={}: {:.2%}'.format(len(Y_test), np.mean(Y_test_pred == Y_test)))