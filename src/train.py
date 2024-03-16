
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import torch
import torch.nn as nn
import torch.nn.functional as F


import os
import os.path as osp
import time
import datetime
import numpy as np
import shutil
from torch import optim
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter


import warnings # ignore warnings
warnings.filterwarnings("ignore")

from dataloader import load_data
from model import load_model
from config.config_util import get_cfg
from util.misc import evaluate, update_best_model, save_cfg_and_args, load_weight, \
    init_seeds, print_args, print_yml_cfg, print_time

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Classification", add_help=add_help)

    parser.add_argument('--data_name', default='cifar-10')
    parser.add_argument('--model_name', default='resnet50')
    parser.add_argument('--mu_threshold', type=float, default=None)
    parser.add_argument('--load_baseline', action='store_true')
    parser.add_argument('--baseline_model_path', default='')
    parser.add_argument('--lr', type=float, default='1e-2')
    parser.add_argument('--epochs', type=int, default='200')
    parser.add_argument('--gpu_id', type=str, default='1')
    parser.add_argument('--result_path', type=str, default='./work_dir')
    parser.add_argument('--cfg', type=str, default='one_stage.yml')
    parser.add_argument('--best_model_path', action='store_const', const=None, help="don't modify")
    parser.add_argument('--print_report', action='store_true')
    parser.add_argument('--print_freq', type=int, default='10')
    parser.add_argument('--train_baseline', action='store_true')
    parser.add_argument('--seed', type=int, default=0)


    return parser.parse_args()

def main(args):
    print_args(args)

    init_seeds(args.seed)  # init seed

    # get cfg
    cfg = get_cfg(args.cfg)[args.data_name]
    print_yml_cfg(cfg)

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_name    = args.data_name
    model_name   = args.model_name
    lr           = float(args.lr)
    momentum     = cfg["optimizer"]["momentum"]
    weight_decay = float(cfg["optimizer"]["weight_decay"])
    epochs       = args.epochs

    # set result path
    if args.train_baseline:
        args.result_path = os.path.join(
            args.result_path,
            "baseline",
            f"{data_name}_{model_name}"
        )
    else:
        args.result_path = os.path.join(
            args.result_path,
            f"{data_name}_{model_name}",
            f"{datetime.datetime.now().strftime('%Y%m%d/%H%M%S')}"
        )
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print(f"\n[INFO] result path: {osp.abspath(args.result_path)}\n")

    # save cfg and args
    save_cfg_and_args(args.result_path, cfg, args)
    
    # save code
    code_path = os.path.join(args.result_path, "code.py")
    cur_file_path = __file__
    shutil.copy(cur_file_path, code_path)

    # data loader
    data_loaders = load_data(data_name)
    
    # model
    model = load_model(
        model_name=model_name,
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        mu_threshold=args.mu_threshold
    )
    model.to(device)

    if args.load_baseline:
        print("\n[INFO] load baseline weight")
        load_weight(model, args.baseline_model_path)

    # loss fn
    loss_fn = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # lr scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=epochs
    )
    
    writer = SummaryWriter(args.result_path)

    begin_time = time.time()
    best_acc = 0
    for epoch in range(epochs):
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        print(f"\nEpoch {epoch}")
        print(f"lr is: {cur_lr}\n")

        # start train
        train_loss, train_acc = train_one_epoch(data_loaders["train"], model, loss_fn, optimizer, device,
                        print_report=args.print_report, print_freq=args.print_freq)
        
        # update the learning rate
        scheduler.step()

        val_acc = evaluate(data_loaders["val"], model, device, print_report=args.print_report)
        
        # log
        tag_scalar_dict = {
            "train loss": train_loss,
            "train acc": train_acc,
            "test acc": val_acc,
        }
        writer.add_scalars(main_tag="loss & acc", tag_scalar_dict=tag_scalar_dict, global_step=epoch)

        model_state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc': best_acc,
        }
        torch.save(model_state, os.path.join(args.result_path, "latest.pth"))

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"\n[FEAT] best acc: {best_acc:.4f}, error rate: {(1 - best_acc):.4f}")
            model_name=f"best-model-acc{best_acc:.4f}.pth"
            update_best_model(args, model_state, model_name)

    writer.close()
    print("Done!")
    print(f"\n[INFO] best acc: {best_acc:.4f}, error rate: {(1 - best_acc):.4f}\n")
    print_time(time.time()-begin_time)


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
            pred = model(X)

            loss = loss_fn(pred, y)
            train_loss += loss.item()

            y_pred_list.extend(pred.argmax(1).cpu().numpy())
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if batch % print_freq == 0:
            print(f"train | loss: {loss.item():>7f}", flush=True)
    
    train_loss /= num_batches
    correct = accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    
    print(f"\n[INFO] Train Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f}\n")

    if print_report:
        print(classification_report(y_true_list, y_pred_list, digits=4))
        
    return train_loss, correct
    

if __name__ == "__main__":
    args = get_args_parser()

    main(args)
    