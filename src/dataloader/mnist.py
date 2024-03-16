from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist(batch_size=128, data_root='data', train=True, val=True, **kwargs):
    ds = {}
    if train:
        train_loader = DataLoader(
            datasets.MNIST(
                root=data_root,
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds["train"] = train_loader

    if val:
        test_loader = DataLoader(
            datasets.MNIST(
                root=data_root,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds["val"] = test_loader

    ds = ds[0] if len(ds) == 1 else ds
    return ds
