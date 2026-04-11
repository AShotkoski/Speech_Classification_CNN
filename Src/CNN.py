import torch
import torch.nn as nn


class net(nn.Module):
    """cnn-trad-fpool3 from Sainath & Parada (2015)
    'Convolutional Neural Networks for Small-footprint Keyword Spotting'
    """
    def __init__(self, num_classes, dropout=0.3):
        super(net, self).__init__()

        # Conv1: 128 filters, 20(time) x 8(freq)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(20, 8))
        self.bn1 = nn.BatchNorm2d(128)

        # Frequency-domain max pooling with pool width 3
        self.pool = nn.MaxPool2d(kernel_size=(1, 3))

        # Conv2: 128 filters, 10(time) x 4(freq)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(10, 4))
        self.bn2 = nn.BatchNorm2d(128)

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(128, 128)
        self.dnn = nn.Linear(128, 256)
        self.output = nn.Linear(256, num_classes)

    def forward(self, x):
        # (B, n_mels, T) -> (B, 1, n_mels, T)
        x = x.unsqueeze(1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2(self.conv2(x)))

        # Global average pool over remaining spatial dims to handle variable sizes
        x = x.mean(dim=(-2, -1))

        x = self.dropout(x)
        x = torch.relu(self.linear(x))
        x = self.dropout(x)
        x = torch.relu(self.dnn(x))
        x = self.output(x)
        return x