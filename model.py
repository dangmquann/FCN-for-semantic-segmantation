import torch.nn as nn
import torchvision.models as models
import torch

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

class FCN_Resnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load pre-trained Resnet18 model
        self.resnet = models.resnet18(pretrained=True)

        # Remove the last 2 layers (avgpool and fc)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Add a 1x1 convolutional layer
        self.conv1x1 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Add a transposed convolutional layer
        self.transposed_conv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32)

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv1x1(x)
        x = self.transposed_conv(x)
        return x
