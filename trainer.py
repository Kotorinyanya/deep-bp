import os.path as osp
from itertools import repeat

import sklearn
from functools import partial
import random

import torch
import torch.distributed as dist
from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch
from boxx import timeit
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from torch_geometric.datasets import TUDataset

from utils import *
import time
from contextlib import closing
import socket

import numpy as np

from dataset import BAvsER, SBM4
from models.classification import Net
from models.community_detection import CD_BP_Net, CD_GCN_Net


def to_cuda(data_list, device):
    for i, data in enumerate(data_list):
        for k, v in data:
            data[k] = v.to(device)
        data_list[i] = data
    return data_list


def find_open_port():
    for port in range(10000, 30000):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            res = sock.connect_ex(('localhost', port))
            if res == 111:  # not in use
                return port


def train_cross_validation(model_cls, dataset, num_clusters, dropout=0.0, lr=1e-4,
                           weight_decay=1e-2, num_epochs=200, n_splits=10,
                           use_gpu=True, dp=False, ddp=True,
                           comment='', tb_service_loc='192.168.192.57:6006', batch_size=1,
                           num_workers=0, pin_memory=False, cuda_device=None,
                           fold_no=None, saved_model_path=None,
                           device_ids=None, patience=50, seed=None, save_model=True,
                           c_reg=0, base_log_dir='runs', base_model_save_dir='saved_models'):
    """
    :param c_reg:
    :param save_model: bool
    :param seed:
    :param patience: for early stopping
    :param device_ids: for ddp
    :param saved_model_path:
    :param fold_no:
    :param ddp: DDP
    :param cuda_device:
    :param pin_memory: DataLoader args https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    :param num_workers: DataLoader args
    :param model_cls: pytorch Module cls
    :param dataset: pytorch Dataset cls
    :param dropout:
    :param lr:
    :param weight_decay:
    :param num_epochs:
    :param n_splits: number of kFolds
    :param use_gpu: bool
    :param dp: bool
    :param comment: comment in the logs, to filter runs in tensorboard
    :param tb_service_loc: tensorboard service location
    :param batch_size: Dataset args not DataLoader
    :return:
    """
    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)

    if ddp and not torch.distributed.is_initialized():  # initialize ddp
        dist.init_process_group('nccl', init_method='tcp://localhost:{}'.format(find_open_port()), world_size=1, rank=0)

    model_name = model_cls.__name__

    if not cuda_device:
        if device_ids and (ddp or dp):
            device = device_ids[0]
        else:
            device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    else:
        device = cuda_device

    device_count = torch.cuda.device_count() if dp else 1
    device_count = len(device_ids) if (device_ids is not None and (dp or ddp)) else device_count
    if device_count > 1:
        print("Let's use", device_count, "GPUs!")

    # batch_size = batch_size * device_count

    log_dir_base = get_model_log_dir(comment, model_name)
    if tb_service_loc is not None:
        print("TensorBoard available at http://{1}/#scalars&regexInput={0}".format(
            log_dir_base, tb_service_loc))
    else:
        print("Please set up TensorBoard")

    criterion = nn.CrossEntropyLoss()

    # get test set
    folds = StratifiedKFold(n_splits=n_splits, shuffle=False)
    train_val_idx, test_idx = list(folds.split(np.zeros(len(dataset)), dataset.data.y.numpy()))[0]
    test_dataset = dataset.__indexing__(test_idx)
    train_val_dataset = dataset.__indexing__(train_val_idx)

    print("Training {0} {1} models for cross validation...".format(n_splits, model_name))
    # folds, fold = KFold(n_splits=n_splits, shuffle=False, random_state=seed), 0
    folds = StratifiedKFold(n_splits=n_splits, shuffle=False)
    iter = folds.split(np.zeros(len(train_val_dataset)), train_val_dataset.data.y.numpy())
    fold = 0

    for train_idx, val_idx in tqdm_notebook(iter, desc='CV', leave=False):

        fold += 1
        if fold_no is not None:
            if fold != fold_no:
                continue

        writer = SummaryWriter(log_dir=osp.join(base_log_dir, log_dir_base + str(fold)))
        model_save_dir = osp.join(base_model_save_dir, log_dir_base + str(fold))

        print("creating dataloader tor fold {}".format(fold))

        model = model_cls(writer, num_clusters=num_clusters, in_dim=dataset.data.x.shape[1],
                          out_dim=int(dataset.data.y.max() + 1), dropout=dropout)

        # My Batch

        train_dataset = train_val_dataset.__indexing__(train_idx)
        val_dataset = train_val_dataset.__indexing__(val_idx)

        train_dataset = dataset_gather(train_dataset, seed=0, n_repeat=1,
                                       n_splits=int(len(train_dataset) / batch_size) + 1)
        val_dataset = dataset_gather(val_dataset, seed=0, n_repeat=1,
                                      n_splits=int(len(val_dataset) / batch_size) + 1)

        train_dataloader = DataLoader(train_dataset,
                                      shuffle=True,
                                      batch_size=device_count,
                                      collate_fn=lambda data_list: data_list,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
        val_dataloader = DataLoader(val_dataset,
                                     shuffle=False,
                                     batch_size=device_count,
                                     collate_fn=lambda data_list: data_list,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)

        # if fold == 1 or fold_no is not None:
        print(model)
        writer.add_text('model_summary', model.__repr__())
        writer.add_text('training_args', str(saved_args))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if ddp:
            model = model.cuda() if device_ids is None else model.to(device_ids[0])
            model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        elif dp and use_gpu:
            model = model.cuda() if device_ids is None else model.to(device_ids[0])
            model = DataParallel(model, device_ids=device_ids)
        elif use_gpu:
            model = model.to(device)

        if saved_model_path is not None:
            model.load_state_dict(torch.load(saved_model_path))

        best_map, patience_counter, best_score = 0.0, 0, -np.inf
        for epoch in tqdm_notebook(range(1, num_epochs + 1), desc='Epoch', leave=False):

            for phase in ['train', 'validation']:

                if phase == 'train':
                    model.train()
                    dataloader = train_dataloader
                else:
                    model.eval()
                    dataloader = val_dataloader

                # Logging
                running_total_loss = 0.0
                running_corrects = 0
                running_reg_loss = 0.0
                running_nll_loss = 0.0
                epoch_yhat_0, epoch_yhat_1 = torch.tensor([]), torch.tensor([])
                epoch_label, epoch_predicted = torch.tensor([]), torch.tensor([])

                for data_list in tqdm_notebook(dataloader, desc=phase, leave=False):

                    # TODO: check devices
                    if dp:
                        data_list = to_cuda(data_list, (device_ids[0]
                                                        if device_ids is not None else
                                                        'cuda'))

                    y_hat, reg = model(data_list)
                    # y_hat = y_hat.reshape(batch_size, -1)

                    y = torch.tensor([], dtype=dataset.data.y.dtype, device=device)
                    for data in data_list:
                        y = torch.cat([y, data.y.view(-1).to(device)])

                    loss = criterion(y_hat, y)
                    reg_loss = -reg
                    total_loss = (loss + reg_loss * c_reg).sum()

                    if phase == 'train':
                        # print(torch.autograd.grad(y_hat.sum(), model.saved_x, retain_graph=True))
                        optimizer.zero_grad()
                        total_loss.backward(retain_graph=True)
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()

                    _, predicted = torch.max(y_hat, 1)
                    label = y

                    running_nll_loss += loss.item()
                    running_total_loss += total_loss.item()
                    running_reg_loss += reg.sum().item()
                    running_corrects += (predicted == label).sum().item()

                    epoch_yhat_0 = torch.cat([epoch_yhat_0, y_hat[:, 0].detach().view(-1).cpu()])
                    epoch_yhat_1 = torch.cat([epoch_yhat_1, y_hat[:, 1].detach().view(-1).cpu()])
                    epoch_label = torch.cat([epoch_label, label.detach().cpu().float()])
                    epoch_predicted = torch.cat([epoch_predicted, predicted.detach().cpu().float()])

                # precision = sklearn.metrics.precision_score(epoch_label, epoch_predicted, average='micro')
                # recall = sklearn.metrics.recall_score(epoch_label, epoch_predicted, average='micro')
                # f1_score = sklearn.metrics.f1_score(epoch_label, epoch_predicted, average='micro')
                accuracy = sklearn.metrics.accuracy_score(epoch_label, epoch_predicted)
                epoch_total_loss = running_total_loss / dataloader.__len__()
                epoch_nll_loss = running_nll_loss / dataloader.__len__()
                epoch_reg_loss = running_reg_loss / dataloader.dataset.__len__()

                writer.add_scalars('nll_loss',
                                   {'{}_nll_loss'.format(phase): epoch_nll_loss},
                                   epoch)
                writer.add_scalars('accuracy',
                                   {'{}_accuracy'.format(phase): accuracy},
                                   epoch)
                # writer.add_scalars('{}_APRF'.format(phase),
                #                    {
                #                        'accuracy': accuracy,
                #                        'precision': precision,
                #                        'recall': recall,
                #                        'f1_score': f1_score
                #                    },
                #                    epoch)
                if epoch_reg_loss != 0:
                    writer.add_scalars('reg_loss'.format(phase),
                                       {'{}_reg_loss'.format(phase): epoch_reg_loss},
                                       epoch)
                # writer.add_histogram('hist/{}_yhat_0'.format(phase),
                #                      epoch_yhat_0,
                #                      epoch)
                # writer.add_histogram('hist/{}_yhat_1'.format(phase),
                #                      epoch_yhat_1,
                #                      epoch)

                # Save Model & Early Stopping
                if phase == 'validation':
                    model_save_path = model_save_dir + '-{}-{}-{:.3f}-{:.3f}'.format(model_name, epoch, accuracy,
                                                                                     epoch_nll_loss)
                    if accuracy > best_map:
                        best_map = accuracy
                        model_save_path = model_save_path + '-best'

                    score = -epoch_nll_loss
                    if score > best_score:
                        patience_counter = 0
                        best_score = score
                    else:
                        patience_counter += 1

                    # skip 10 epoch
                    # best_score = best_score if epoch > 10 else -np.inf

                    if save_model:
                        for th, pfix in zip([0.8, 0.75, 0.7, 0.5, 0.0],
                                            ['-perfect', '-great', '-good', '-bad', '-miss']):
                            if accuracy >= th:
                                model_save_path += pfix
                                break
                        if epoch > 10:
                            torch.save(model.state_dict(), model_save_path)

                    writer.add_scalars('best_val_accuracy',
                                       {'{}_accuracy'.format(phase): best_map},
                                       epoch)
                    writer.add_scalars('best_nll_loss',
                                       {'{}_nll_loss'.format(phase): -best_score},
                                       epoch)

                    if patience_counter >= patience:
                        print("Stopped at epoch {}".format(epoch))
                        return

    print("Done !")


def train_cummunity_detection(model_cls, dataset, dropout=0.0, lr=1e-3,
                              weight_decay=1e-2, num_epochs=200, n_splits=10,
                              use_gpu=True, dp=False, ddp=False,
                              comment='', tb_service_loc='192.168.192.57:6006', batch_size=1,
                              num_workers=0, pin_memory=False, cuda_device=None,
                              ddp_port='23456', fold_no=None, device_ids=None, patience=20, seed=None,
                              save_model=False, supervised=False):
    """
    :param save_model: bool
    :param seed:
    :param patience: for early stopping
    :param device_ids: for ddp
    :param saved_model_path:
    :param fold_no:
    :param ddp_port:
    :param ddp: DDP
    :param cuda_device:
    :param pin_memory: DataLoader args https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    :param num_workers: DataLoader args
    :param model_cls: pytorch Module cls
    :param dataset: pytorch Dataset cls
    :param dropout:
    :param lr:
    :param weight_decay:
    :param num_epochs:
    :param n_splits: number of kFolds
    :param use_gpu: bool
    :param dp: bool
    :param comment: comment in the logs, to filter runs in tensorboard
    :param tb_service_loc: tensorboard service location
    :param batch_size: Dataset args not DataLoader
    :return:
    """

    saved_args = locals()
    seed = int(time.time() % 1e4 * 1e5) if seed is None else seed
    saved_args['random_seed'] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)

    if ddp and not torch.distributed.is_initialized():  # initialize ddp
        dist.init_process_group('nccl', init_method='tcp://localhost:{}'.format(ddp_port), world_size=1, rank=0)

    model_name = model_cls.__name__

    if not cuda_device:
        if device_ids and (ddp or dp):
            device = device_ids[0]
        else:
            device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    else:
        device = cuda_device

    device_count = torch.cuda.device_count() if dp else 1
    device_count = len(device_ids) if (device_ids is not None and (dp or ddp)) else device_count
    if device_count > 1:
        print("Let's use", device_count, "GPUs!")

    # batch_size = batch_size * device_count

    log_dir_base = get_model_log_dir(comment, model_name)
    if tb_service_loc is not None:
        print("TensorBoard available at http://{1}/#scalars&regexInput={0}".format(
            log_dir_base, tb_service_loc))
    else:
        print("Please set up TensorBoard")

    print("Training {0} {1} models for cross validation...".format(n_splits, model_name))
    folds, fold = KFold(n_splits=n_splits, shuffle=False, random_state=seed), 0
    print(dataset.__len__())

    for train_idx, test_idx in tqdm_notebook(folds.split(list(range(dataset.__len__())),
                                                         list(range(dataset.__len__()))),
                                             desc='models', leave=False):
        fold += 1
        if fold_no is not None:
            if fold != fold_no:
                continue

        writer = SummaryWriter(log_dir=osp.join('runs', log_dir_base + str(fold)))
        model_save_dir = osp.join('saved_models', log_dir_base + str(fold))

        print("creating dataloader tor fold {}".format(fold))

        model = model_cls(writer, dropout=dropout)

        # My Batch
        train_dataset = dataset.__indexing__(train_idx)
        test_dataset = dataset.__indexing__(test_idx)

        train_dataset = dataset_gather(train_dataset, n_repeat=1,
                                       n_splits=int(len(train_dataset) / batch_size))

        train_dataloader = DataLoader(train_dataset,
                                      shuffle=True,
                                      batch_size=device_count,
                                      collate_fn=lambda data_list: data_list,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
        test_dataloader = DataLoader(test_dataset,
                                     shuffle=False,
                                     batch_size=device_count,
                                     collate_fn=lambda data_list: data_list,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)

        print(model)
        writer.add_text('model_summary', model.__repr__())
        writer.add_text('training_args', str(saved_args))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if ddp:
            model = model.cuda() if device_ids is None else model.to(device_ids[0])
            model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
        elif dp and use_gpu:
            model = model.cuda() if device_ids is None else model.to(device_ids[0])
            model = DataParallel(model, device_ids=device_ids)
        elif use_gpu:
            model = model.to(device)

        for epoch in tqdm_notebook(range(1, num_epochs + 1), desc='Epoch', leave=False):

            for phase in ['train', 'validation']:

                if phase == 'train':
                    model.train()
                    dataloader = train_dataloader
                else:
                    model.eval()
                    dataloader = test_dataloader

                # Logging
                running_total_loss = 0.0
                running_reg_loss = 0.0
                running_overlap = 0.0

                for data_list in tqdm_notebook(dataloader, desc=phase, leave=False):

                    # TODO: check devices
                    if dp:
                        data_list = to_cuda(data_list, (device_ids[0]
                                                        if device_ids is not None else
                                                        'cuda'))

                    y_hat, reg = model(data_list)

                    y = torch.tensor([], dtype=dataset.data.y.dtype, device=device)
                    for data in data_list:
                        y = torch.cat([y, data.y.view(-1).to(device)])

                    if supervised:
                        loss = permutation_invariant_loss(y_hat, y)
                        # criterion = nn.NLLLoss()
                        # loss = criterion(y_hat, y)
                    else:
                        loss = -reg
                    total_loss = loss

                    if phase == 'train':
                        # print(torch.autograd.grad(y_hat.sum(), model.saved_x, retain_graph=True))
                        optimizer.zero_grad()
                        total_loss.backward(retain_graph=True)
                        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()

                    _, predicted = torch.max(y_hat, 1)
                    label = y

                    if supervised:
                        overlap_score = normalized_overlap(label.int().cpu().numpy(), predicted.int().cpu().numpy(),
                                                           0.25)
                        # overlap_score = overlap(label.int().cpu().numpy(), predicted.int().cpu().numpy())
                        running_overlap += overlap_score
                        print(reg, overlap_score, loss)

                    running_total_loss += total_loss.item()
                    running_reg_loss += reg.sum().item()

                epoch_total_loss = running_total_loss / dataloader.__len__()
                epoch_reg_loss = running_reg_loss / dataloader.dataset.__len__()
                if supervised:
                    epoch_overlap = running_overlap / dataloader.__len__()
                    writer.add_scalars('overlap'.format(phase),
                                       {'{}_overlap'.format(phase): epoch_overlap},
                                       epoch)

                writer.add_scalars('reg_loss'.format(phase),
                                   {'{}_reg_loss'.format(phase): epoch_reg_loss},
                                   epoch)

    print("Done !")


if __name__ == "__main__":
    # dataset = SBM4(root='datasets/SBM4')
    trans = partial(pad_with_zero, 126)
    dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES', transform=trans)
    dataset.data.edge_attr = None
    # model = CD_BP_Net
    model = Net
    train_cross_validation(model, dataset, 6, comment='bp-debug', batch_size=40, fold_no=2,
                           num_epochs=100, patience=100, dropout=0, lr=1e-4, weight_decay=0,
                           use_gpu=True, dp=False, ddp=True, device_ids=[2], c_reg=0.1, save_model=False)
    # train_cummunity_detection(model, dataset, comment='debug', batch_size=1,
    #                           num_epochs=100, patience=100, dropout=0, lr=1e-2, weight_decay=0,
    #                           use_gpu=True, dp=False, ddp=True, device_ids=[4], supervised=True)
