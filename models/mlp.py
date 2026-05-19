import torch.nn as nn
from utils.seed import set_seed

set_seed(42)
class MLPClassifier(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        num_classes=2
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        return self.net(x)