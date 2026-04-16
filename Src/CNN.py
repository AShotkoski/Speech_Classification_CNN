import torch
import torch.nn as nn
import torchvision.models as models

class net(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(net, self).__init__()
        
        # Initialize EfficientNet B0 architecture
        self.model = models.efficientnet_b0(weights=None)
        
        # Modify the first Convolutional Layer to accept 1 channel (for spectrograms) instead of 3
        # The first layer in torchvision's EfficientNet is located at features[0][0]
        original_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=(original_conv.bias is not None)
        )
        
        # Modify the classifier to match the target number of classes and specify dropout
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Unsqueeze the channel dimension to treat spectrogram sequences as 1-channel data if needed
        x = x.unsqueeze(1)
        x = self.model(x)
        return x