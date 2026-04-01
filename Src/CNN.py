import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3,3),padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(3,3), padding=1),
            torch.nn.ReLU,
            torch.nn.Flatten(),
            torch.nn.Linear(20000,1)
        )

    def forward(self, x):
        out = self.main(x)
        return out