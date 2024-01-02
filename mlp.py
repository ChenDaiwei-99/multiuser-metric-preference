import torch.nn as nn
    
class projector(nn.Module):
    
    def __init__(self, feature_dim=512, num_layer=1, num_class=10, bias_ind=True, latent_size=None) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layer = num_layer
        self.num_class = num_class
        self.latent_size = latent_size
        
        if num_layer == 1:
            self.layers = nn.Linear(self.feature_dim, self.num_class, bias=bias_ind)
        elif num_layer == 2:
            self.fc_first = nn.Linear(self.feature_dim, self.latent_size, bias=bias_ind)
            self.fc_end = nn.Linear(self.latent_size, self.num_class, bias=bias_ind)
            layers = [self.fc_first, nn.ReLU(), self.fc_end]
            self.layers = nn.Sequential(*layers)
        else:
            layers = []
            self.fc_first = nn.Linear(self.feature_dim, self.latent_size, bias=bias_ind)
            self.fc_end = nn.Linear(self.latent_size, self.num_class, bias=bias_ind)
            layers.append(self.fc_first)
            for i in range(num_layer-2):
                layers.append(nn.Linear(self.latent_size, self.latent_size, bias=bias_ind))
                layers.append(nn.ReLU())
            layers.append(self.fc_end)
            self.layers = nn.Sequential(*layers)
        
    def forward(self,x):
        x = self.layers(x)
        return x