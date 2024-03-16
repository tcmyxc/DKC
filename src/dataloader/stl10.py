from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_stl10(batch_size=32, data_root='data', train=True, val=True, **kwargs):
    ds = {}
    if train:
        train_loader = DataLoader(
            datasets.STL10(
                root=data_root, 
                split='train', 
                download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds["train"] = train_loader

    if val:
        test_loader = DataLoader(
            datasets.STL10(
                root=data_root, 
                split='test', 
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds["val"] = test_loader

    ds = ds[0] if len(ds) == 1 else ds
    return ds
