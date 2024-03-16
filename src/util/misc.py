
import os
import json
import math
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from copy import deepcopy

from sklearn.metrics import classification_report, accuracy_score


def evaluate(dataloader, model, device, print_report=True, print_freq=10):
    y_pred_list = []
    y_true_list = []
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y_true_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        y_pred_list.extend(pred.argmax(1).cpu().numpy())

    correct /= size
        
    print(f"\n[INFO] Test Error: Accuracy: {(100*correct):>0.2f}%\n")
    
    if print_report:
        print(classification_report(y_true_list, y_pred_list, digits=4))
    
    return correct


def train_one_epoch(dataloader, model, loss_fn, optimizer, device, print_report=True, print_freq=10):
    y_pred_list = []
    y_true_list = []

    train_loss, correct = 0, 0
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y_true_list.extend(y.numpy())

        X, y = X.to(device), y.to(device)

        with torch.set_grad_enabled(True):
            pred = model(X)  # 网络前向计算

            loss = loss_fn(pred, y)
            train_loss += loss.item()

            y_pred_list.extend(pred.argmax(1).cpu().numpy())
            
            # Backpropagation
            optimizer.zero_grad()  # 清除过往梯度
            loss.backward()  # 得到模型中参数对当前输入的梯度
            optimizer.step()  # 更新参数
        
        if batch % print_freq == 0:
            print(f"train | loss: {loss.item():>7f}", flush=True)
    
    train_loss /= num_batches
    correct = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    
    print(f"\n[INFO] Train Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f}\n")

    if print_report:
        print(classification_report(y_true_list, y_pred_list, digits=4))


def load_weight(model, model_path):
    """加载模型权重"""
    print('\n==> load weight')
    weights_dict = torch.load(model_path, map_location='cpu')["model"]
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict , strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)


def update_best_model(cfg, model_state, model_name):
    """更新权重文件"""

    result_path = cfg.result_path
    cp_path = os.path.join(result_path, model_name)

    if cfg.best_model_path is not None:
        # remove previous model weights
        os.remove(cfg.best_model_path)

    torch.save(model_state, cp_path)
    torch.save(model_state, os.path.join(result_path, "best-model.pth"))
    cfg.best_model_path = cp_path
    print(f"\n[INFO] Saved Best PyTorch Model State to {model_name}\n")


def save_cfg_and_args(result_path, cfg=None, args=None):
    '''保存 cfg 和 args 到文件'''

    if args is not None:
        with open(os.path.join(result_path, 'args.json'), 'w') as f:
            json.dump(vars(args), f)

    if cfg is not None:
        with open(os.path.join(result_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f)

def init_seeds(seed=0):
    """固定随机种子"""
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def print_args(args):
    """优雅地打印命令行参数"""
    
    print("")
    print("-" * 20, "args", "-" * 20)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 18, "args end", "-" * 18, flush=True)

def print_yml_cfg(cfg):
    """打印从yml文件加载的配置"""

    print("")
    print("-" * 20, "yml cfg", "-" * 20)
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("-" * 18, "yml cfg end", "-" * 18, flush=True)

def get_current_time():
    '''get current time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
    return f"{ymd}_{hms}"


def print_time(time_elapsed, epoch=False):
    """打印程序执行时长"""
    time_hour = time_elapsed // 3600
    time_minite = (time_elapsed % 3600) // 60
    time_second = time_elapsed % 60
    if epoch:
        print(f"\nCurrent epoch take time: {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")
    else:
        print(f"\nAll complete in {time_hour:.0f}h {time_minite:.0f}m {time_second:.0f}s")

      
def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
