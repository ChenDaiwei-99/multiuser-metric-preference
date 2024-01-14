import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MetricPrefLearner(nn.Module):
    
    def __init__(self, dim_feature, num_users, items):
        '''
        items shape: np.array (dim_feature, num_items)
        '''
        super().__init__()
        self.dim_feature = dim_feature
        self.num_users = num_users
        self.items = items
        self.us = nn.Parameter(torch.randn((self.dim_feature, self.num_users)))

    def assignModel(self, init_model):
        '''
        Use whatever model you like
        One Condition must be satisfied: input_dim == output_dim == dim_feature
        '''
        self.model = init_model

    def forward(self, x):
        '''
        x shape: (user_id, (item_i, item_j))
        '''
        user_ids, x_is, x_js = x[0], x[1][0], x[1][1]
        x_is, x_js = torch.Tensor(self.items[:,x_is]), torch.Tensor(self.items[:,x_js])
        us_k_track =self.us[:,user_ids]
        x_is_minus_us = (x_is - us_k_track).T
        x_js_minus_us = (x_js - us_k_track).T
        ele_1 = self.model(x_is_minus_us)
        ele_2 = self.model(x_js_minus_us)
        delta = torch.sum(ele_1 * ele_1, dim=1) - torch.sum(ele_2 * ele_2, dim=1)
        return delta
    
class MetricPrefLearner_groups(nn.Module):
    
    def __init__(self, dim_feature, num_users, items, num_groups=None):
        '''
        items shape: np.array (dim_feature, num_items)
        '''
        super().__init__()
        self.dim_feature = dim_feature
        self.num_users = num_users
        self.items = items
        self.num_groups = num_groups
        # assert num_groups != 1, 'number of groups should larger than 1'

        # two different modeling methods (depend on the num_groups)
        if self.num_groups == None:
            # self.us = nn.Parameter(torch.randn((self.dim_feature, self.num_users)))
            self.us = nn.Parameter(
                torch.tensor(np.random.multivariate_normal(np.zeros(self.dim_feature),(1/self.dim_feature)*np.eye(self.dim_feature),self.num_users).T).float()
            )
        elif self.num_groups >= 1:
            # self.us_groups = nn.Parameter(torch.randn((self.dim_feature, self.num_groups)))
            self.us_groups = nn.Parameter(
                torch.tensor(np.random.multivariate_normal(np.zeros(self.dim_feature),(1/self.dim_feature)*np.eye(self.dim_feature),self.num_groups).T).float()
            )
            self.unconstrained_weights = nn.Parameter(torch.randn((self.num_groups, self.num_users)))   # this is unconstrained weights, need to use softmax function to normalize it into probabilities
            self.softmax = nn.Softmax(dim=0)    # alongside the num_groups dimension

    def assignModel(self, init_model):
        '''
        Use whatever model you like
        One Condition must be satisfied: input_dim == output_dim == dim_feature
        '''
        self.model = init_model

    def forward(self, x):
        '''
        x shape: (user_id, (item_i, item_j))
        '''

        user_ids, x_is, x_js = x[0], x[1][0], x[1][1]
        x_is, x_js = torch.Tensor(self.items[:,x_is]), torch.Tensor(self.items[:,x_js])
        if self.num_groups == None:
            us_k_track =self.us[:,user_ids]
        else:
            us_probs = self.softmax(self.unconstrained_weights[:,user_ids])
            # print(self.us_groups)
            us_k_track = self.us_groups @ us_probs
        x_is_minus_us = (x_is - us_k_track).T
        x_js_minus_us = (x_js - us_k_track).T
        ele_1 = self.model(x_is_minus_us)
        ele_2 = self.model(x_js_minus_us)
        delta = torch.sum(ele_1 * ele_1, dim=1) - torch.sum(ele_2 * ele_2, dim=1)
        return delta
