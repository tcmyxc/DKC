import torch.nn as nn
import torch.nn.functional as F
from model.kgl import KernelGaussianLayer


class SimpleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, mu_threshold=0.5):
        super(SimpleNet, self).__init__()
        self.features = self._make_layers(in_channels)
        # add
        self.kgl = KernelGaussianLayer(in_channels=256, mu_threshold=mu_threshold)
        self.classifier = nn.Linear(256, num_classes)
        self.drp = nn.Dropout(0.1)

    def forward(self, x):
        out = self.features(x)

        # Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = self.drp(out)
        out = out.view(out.size(0), -1)
        out = self.kgl(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_channels):

        model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(512, 2048, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(2048, 256, kernel_size=[1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

        )

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model


def simplenet(in_channels=3, num_classes=10, mu_threshold=0.5):
    return SimpleNet(in_channels=in_channels, num_classes=num_classes, mu_threshold=mu_threshold)

if __name__ == "__main__":
    from torchsummary import summary
    model = simplenet(mu_threshold=0.2)
    print(model)
    
    summary(model, (3, 32, 32), device="cpu")
