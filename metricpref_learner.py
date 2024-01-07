import torch
import torch.nn as nn
import torch.nn.functional as F

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
