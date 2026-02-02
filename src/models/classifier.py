import torch.nn as nn

class LinearClassifier(nn.Module):
    """Linear classifier for evaluation."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class MLPClassifier(nn.Module):
    """MLP classifier with one hidden layer."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
        self.num_classes = num_classes

    def reset_parameters(self):
        self.fc.reset_parameters()
