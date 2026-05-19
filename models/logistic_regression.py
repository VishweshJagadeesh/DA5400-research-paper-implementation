import torch.nn as nn
from utils.seed import set_seed

set_seed(42)
class LogisticRegression(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes
    ):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            num_classes
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        return self.linear(x)