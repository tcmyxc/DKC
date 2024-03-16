from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
    
from dataloader.cifar import get10, get100
from dataloader.stl10 import get_stl10
from dataloader.tiny_imagenet import get_tinyimagenet
from dataloader.mini_imagenet import get_mini
from dataloader.mnist import get_mnist
from dataloader.fashion_mnist import get_fashion_mnist

def load_data(data_name, data_type=None):
    print('\n[INFO] load data:', data_name)

    if data_type is None:
        if data_name == 'cifar-10':
            return get10(batch_size=128)
        elif data_name == 'cifar-100':
            return get100(batch_size=128)
        elif data_name == 'stl-10':
            return get_stl10(batch_size=32)
        elif data_name == 'mini-imagenet':
            return get_mini(batch_size=32)
        elif data_name == 'tiny-imagenet':
            return get_tinyimagenet(batch_size=32)
        elif data_name == 'mnist':
            return get_mnist(batch_size=128)
        elif data_name == 'fashion-mnist':
            return get_fashion_mnist(batch_size=128)


if __name__ == "__main__":
    data_loaders = load_data("tiny-imagenet")
    
    print(len(data_loaders["train"].dataset))
    # print(data_loaders["train"].dataset[0][0].max())
    print(len(data_loaders["val"].dataset))
    # print(data_loaders["val"].dataset[0])
    