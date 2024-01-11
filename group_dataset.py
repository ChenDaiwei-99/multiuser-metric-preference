# Defines dataset object

import numpy as np
import scipy.stats as st
import utils
from sklearn.model_selection import train_test_split
import scipy.io as sio

class Dataset:
    """
    Dataset object to generate and load datasets in unified format

    High level notation guideline, unified across all datasets:
        X: items
        S: comparisons
        Y: responses

    Methods (all after initialization)
        getAllData: returns full dataset in dict
        getTrainTestSplit: returns train and test split of dataset

    Utilities:
        _normal_init: intialize normally distributed data
        _color_init: initialize color dataset

    """
    def __init__(self, dataset_type, **kwargs):
        """
        Principal initialization operation is to load and format the following:
        X: d (ambient dimension) x n (number of items) matrix of item vectors (one item per column)
        S: # comparisons list of tuples. The first tuple entry is the user index. The second tuple entry is another
           tuple containing the first and second pair item indices. e.g., (k, (i,j))
        Y: # comparisons list of responses (-1 for first item, +1 for second item)

        Secondary initialization (if applicable):
        M: ground-truth metric
        U: ground-truth user points

        Required inputs:
            dataset_type: type of dataset, from {'Normal', 'Color'}

        Optional inputs (see defaults in dataset_types dict):
            Normally distributed data:
                d: ambient dimension
                r: metric rank
                n: number of items
                N: number of users
                m: number of measurements per user
                noise_type: type of noise in measurement model, from {'none', 'logistic'}
                noise_param: noise parameter, if relevant (in 'logistic' noise, is noise scaling parameter)
                X: X: if None, generates new data. Else, specifies item matrix

            Color data:
                color_path: path to color data
                N: number of users to subsample uniformly at random (<= 48, None for default of 48)
        """


        # Defaults dictionary. key-value template: {dataset_type: (self._INITIALIZATION_FUNCTION, {DEFAULTS_DICT})}
        #    DEFAULTS_DICT *must* match inputs to self._INITIALIZATION_FUNCTION
        dataset_types = {'Normal': (self._normal_init, {'d': 2, 'r': 2, 'n': 100,
                                   'N': 3, 'm':30, 'noise_type':'none', 'noise_param':None,
                                   'X':None}),
                         'Color': (self._color_init, {'color_path':'./CPdata.mat', 'N':None})}
        self.init_fun, defaults = dataset_types[dataset_type]

        # set parameters from defaults and inputs
        pm = {}
        for key in defaults.keys():
            if key in kwargs:
                pm[key] = kwargs[key]
            else:
                pm[key] = defaults[key]

        # initialize variables
        self.dataset_type = dataset_type
        self.pm = pm # record parameters
        self.raw = None # raw data, if any
        self.X = None
        self.S = None
        self.Y = None
        self.M = None # ground-truth metric is None, unless it exists in which case it is overridden
        self.U = None # ground-truth user points matrix is None, unless they exist in which case they are overridden

        # domain specific initializaion
        self.init_fun(**pm)

    def getAllData(self):
        # returns dict containing X, S, Y, M, U
        return {'X':self.X, 'S':self.S, 'Y':self.Y, 'M':self.M, 'U':self.U}

    def getTrainTestSplit(self, train_size=0.75, blocked=True):
        """
        Creates random train and test split from data

        Inputs:
            train_size (float or int): split fraction or absolute size for training set.
                If int, specifies exact number of training examples (total across all users)
            blocked: flag to block sampling within users (i.e., keep sampling fraction constant within each user)

        Returns (in tuple):
            S_train: training comparisons (in format of S)
            S_test: test comparisons (in format of S),
            Y_train: training responses (in format of Y)
            Y_test: test responses (in format of Y)
        """

        if blocked:
            stratify = [s[0] for s in self.S] # stratify by user indices

        else:
            stratify = None

        S_train, S_test, Y_train, Y_test = train_test_split(self.S, self.Y,
                train_size=train_size, shuffle=True, stratify=stratify)

        return S_train, S_test, Y_train, Y_test

    def _color_init(self, color_path, N):
        """
        Loads and initializes color dataset

        Inputs:
            color_path: path to color data
            N: number of users to subsample uniformly at random (<= 48, None for default of 48)
        """

        # hardcoded parameters
        self.pm['d'] = 3 # CIELAB space
        self.pm['n'] = 37 # number of colors

        # set subsampling, if any
        if N is None:
            self.pm['N'] = 48
        else:
            assert N <= 48
            self.pm['N'] = N

        # load color data
        color_data = sio.loadmat(color_path)
        SingPrefAFCload = color_data['SingPrefAFCload']
        assert SingPrefAFCload.shape[0] == self.pm['n'] and \
            SingPrefAFCload.shape[1] == self.pm['n'] # sanity check on number of colors
        assert SingPrefAFCload.shape[2] == 48 # sanity check on maximum number of users

        # downsample users if needed
        if self.pm['N'] < SingPrefAFCload.shape[2]:
            user_ids = np.random.choice(SingPrefAFCload.shape[2], self.pm['N'], replace=False)
            SingPrefAFCload = SingPrefAFCload[:, :, user_ids]
        else:
            user_ids = list(range(SingPrefAFCload.shape[2]))

        CIELAB = color_data['CIELAB']

        # translate items into universal format
        X = CIELAB.T # 3 x 37 item matrix

        # center and scale data
        X = X - np.mean(X, axis=1)[:, None]
        max_x_norm = max(np.linalg.norm(X, axis=0))
        X = X / max_x_norm

        assert X.shape[0] == self.pm['d'] # sanity check on dimension
        assert X.shape[1] == self.pm['n'] # sanity check on number of items

        S = []
        Y = []
        
        for k in range(self.pm['N']):
            SingPrefAFCload_k = SingPrefAFCload[:, :, k]

            for i in range(self.pm['n']):
                for j in range(self.pm['n']):

                    if i != j:
                        S.append((k, (i, j)))
                        Y.append(SingPrefAFCload_k[i, j] * 2 - 3) # map (1,2) to (-1, 1)

        Y = np.array(Y)

        # save dataset
        self.X = X
        self.S = S
        self.Y = Y
        self.raw = {'SingPrefAFCload':SingPrefAFCload, 'user_ids':user_ids}
        
    def _normal_init(self, d, r, n, N, m, noise_type, noise_param, X=None, num_groups=1):
        '''
        Desc: Generates normally distributed data along with ground-truth metric and user points (Refer to the original code)

        Inputs:
            feature_dim: feature dimensions
            metric_rank: metric rank
            num_items: number of items
            num_users: number of users
            num_pairs_per_user: number of measurements per user
            noise type: type of noise in measurement model, from {'none', 'logistic'}
            noise_beta: noise level
            num_groups: number of subgroups
        '''

        feature_dim = d
        metric_rank = r
        num_items = n
        num_users = N
        num_pairs_per_user = m
        noise_type = noise_type
        noise_beta = noise_param
        num_groups = num_groups

        assert metric_rank <= feature_dim, 'metric rank must be equal to or smaller then feature dim'

        # generate ground-truth metric
        if metric_rank < feature_dim:
            # low-dimensional orthogonal matrix
            L = st.ortho_group.rvs(dim=feature_dim)
            L = L[:,:metric_rank]
            M = (feature_dim / np.sqrt(metric_rank)) * L @ L.T  # keep the frob_norm = feature_dim
        elif metric_rank == feature_dim:
            # arbitrary PSD metric, normalized to have Frobenius norm of d
            L = np.random.multivariate_normal(np.zeros(feature_dim), np.eye(feature_dim), feature_dim)
            M = L @ L.T
            M = M * (feature_dim / np.linalg.norm(M, 'fro'))
        self.M = M
        
        # generate user points (methods depend on the size of the num_groups)
        if num_groups == 1:
            # original code
            U = np.random.multivariate_normal(np.zeros(feature_dim), (1/feature_dim)*np.eye(feature_dim), num_users).T
            self.U = U
            # pseudo user points
            V = -2*M @ U
            # generate items and comparisons
            Xdata = np.random.multivariate_normal(np.zeros(feature_dim), (1/feature_dim)*np.eye(feature_dim), num_items).T
            S = list(zip(list(np.repeat(list(range(num_users)), num_pairs_per_user)),
                            [tuple(np.random.choice(num_items, 2, replace=False)) for _ in range(num_pairs_per_user*num_users)]))
            Y, Y_noiseless, Y_unquant = utils.one_bit_pairs(Xdata, S, M, V, noise_type, noise_beta)
        elif num_groups > 1:
            # only need to create several ideal points (num_groups)
            U_subgroups = np.random.multivariate_normal(np.zeros(feature_dim), (1/feature_dim)*np.eye(feature_dim), num_groups).T
            self.U_subgroups = U_subgroups
            # Then we need to simulate different users, each user's point is a weight-average of the ideal points
            alpha = np.random.rand(num_users, num_groups)
            alpha = alpha / alpha.sum(axis=1).reshape(-1,1)
            self.alpha = alpha
            U = U_subgroups @ alpha.T
            self.U = U
            # pseudo user points
            V = -2*M @ U
            # generate items and comparisons
            Xdata = np.random.multivariate_normal(np.zeros(feature_dim), (1/feature_dim)*np.eye(feature_dim), num_items).T
            S = list(zip(list(np.repeat(list(range(num_users)), num_pairs_per_user)),
                            [tuple(np.random.choice(num_items, 2, replace=False)) for _ in range(num_pairs_per_user*num_users)]))
            Y, Y_noiseless, Y_unquant = utils.one_bit_pairs(Xdata, S, M, V, noise_type, noise_beta)

        # save dataset
        self.X = Xdata
        self.S = S
        self.Y = Y
        self.Y_noiseless = Y_noiseless
        self.Y_unquant = Y_unquant


if __name__ == '__main__':

    ##### Example: normally distributed data. Rather than d=2, r=2 as is default,
    # we will do d=5 dimensions and r=3 rank, and keep the remaining defaults.

    # initialize dataset object
    normal_dataset = Dataset('Normal', d=5, r=3)

    # get items, metric and user points (we don't need the full dataset S and Y)
    data = normal_dataset.getAllData()
    X, M, U = data['X'], data['M'], data['U']

    # get train test split
    S_train, S_test, Y_train, Y_test = normal_dataset.getTrainTestSplit()
    
    ##### Example: color data. Let's subsample 10 users
    colordata = Dataset('Color', N=10)

    # get items
    data = colordata.getAllData()
    X = data['X']

    # get train test split
    S_train, S_test, Y_train, Y_test = colordata.getTrainTestSplit(train_size=500)
