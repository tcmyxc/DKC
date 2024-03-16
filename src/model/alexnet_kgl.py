import torch
import torch.nn as nn
from model.kgl import KernelGaussianLayer


# no LRN
class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, mu_threshold=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # add kgl
        self.kgl = KernelGaussianLayer(in_channels=256, mu_threshold=mu_threshold)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # add
        x = self.kgl(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(in_channels=3, num_classes=10, **kwargs):
    return AlexNet(in_channels=in_channels, num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary
    model = alexnet()
    
    summary(model, (3, 32, 32), batch_size=128, device="cpu")
