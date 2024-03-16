from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get10(batch_size=128, data_root='data', train=True, val=True, **kwargs):
    ds = {}
    if train:
        train_loader = DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds["train"] = train_loader

    if val:
        test_loader = DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds["val"] = test_loader

    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get100(batch_size, data_root='data', train=True, val=True, **kwargs):
    ds = {}
    if train:
        train_loader = DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds["train"] = train_loader

    if val:
        test_loader = DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds["val"] = test_loader

    ds = ds[0] if len(ds) == 1 else ds
    return ds
