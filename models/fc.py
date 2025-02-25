import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLPSmall(nn.Module):
    def __init__(self, input_size, n_bands, n_classes, p_drop = 0.3, *args, **kwargs):

        super().__init__()
        
        fan_in_input = input_size*input_size*n_bands

        fan_out_first = int(2 ** np.floor(np.log2(fan_in_input)))
        fan_out_second = int(2 ** np.floor(np.log2(fan_in_input) - 1))

        print(fan_out_first, fan_out_second)


        self.fc1 = nn.Linear(fan_in_input, fan_out_first)
        self.fc2 = nn.Linear(fan_out_first, fan_out_second)
        self.fc3 = nn.Linear(fan_out_second, n_classes)

        self.dropout = nn.Dropout(p_drop)
        
        
    def forward(self,x):
        # flatten image input
        _, c, h, w = x.shape
        x = x.view(-1, c*h*w) # [batch, c*h*w]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
                
        return x
    


class ArticleMLP(nn.Module):
    def __init__(self, input_size, n_bands, n_classes, p_drop = 0.3, *args, **kwargs):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size*input_size*n_bands, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512, 512)

        self.dropout = nn.Dropout(p_drop)
        
        # set n_class to 0 if we want headless model
        self.n_class = n_classes
        if n_classes:
            self.head = nn.Sequential(
                                  nn.Linear(512, 1024),
                                  nn.ReLU(),
                                  nn.Dropout(p = p_drop),
                                  nn.Linear(1024, n_classes)
            )
        
        
    def forward(self,x):
        # flatten image input
        _, c, h, w = x.shape
        x = x.view(-1, c*h*w) # [batch, c*h*w]
        
        x = F.relu(self.fc1(x)) # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc2(x)) # [batch, 512]
        x = self.dropout(x)
        x = F.relu(self.fc3(x)) # [batch, 512]
        
        if self.n_class:
            x = self.head(x)
        
        return x
