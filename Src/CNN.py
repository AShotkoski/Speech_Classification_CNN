import torch

class net(torch.nn.Module):
    def __init__(self, num_classes):
        super(net, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.main(x)
        return out