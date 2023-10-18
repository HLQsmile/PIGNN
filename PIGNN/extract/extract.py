import os
import torch
from torch import nn
import numpy as np
import torch as t
import torch.nn.functional as F
import scipy.io as sio
from sklearn.metrics import precision_score, f1_score
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from sklearn.model_selection import StratifiedKFold
import argparse
import warnings
warnings.filterwarnings("ignore")
from dataset import (collate_func, DataLoader, ExprDataset)
from scheduler import CosineAnnealingWarmRestarts
from model import CA_SAGE

pathjoin = os.path.join
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--infile', type=str,default=r'..\dataset.npz' )
    parser.add_argument('-output', '--outdir', type=str, default=r'\rusult')
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    return parser

def getfeature(net,input_data):
    x, edge_index, batch = input_data.x, input_data.edge_index, input_data.batch

    if net.weight_edge_flag:
        one_graph_edge_weight = torch.sigmoid(net.edge_weight)  # *self.edge_num
        edge_weight = one_graph_edge_weight
    else:
        edge_weight = None

    x = net.act1(net.conv1(x.to(device), edge_index.to(device), edge_weight=edge_weight.to(device)))
    x = x.permute(2, 0, 1)
    x = x.permute(1, 0, 2)
    x = net.bn1(x)
    x = x.permute(1, 0, 2)
    if net.dropout_ratio > 0: x = F.dropout(x, p=0.1, training=net.training)
    x = x.permute(1, 2, 0)
    x = x.unsqueeze(dim=-1)
    x = net.global_conv1(x)
    x = net.global_act1(x)
    x = net.global_bn1(x)
    if net.dropout_ratio > 0: x = F.dropout(x, p=0.3, training=net.training)
    x = net.global_conv2(x)
    x = net.global_act1(x)
    x = net.global_bn2(x)
    if net.dropout_ratio > 0: x = F.dropout(x, p=0.3, training=net.training)
    x = x.squeeze(dim=-1)
    num_samples = x.shape[0]
    x = x .view(num_samples,-1)
    x = net.global_fc_nn(x)
    return x

def train2(model, optimizer, train_loader, epoch, device, loss_fn=None, scheduler=None, verbose=False):
    model.train()

    loss_all = 0
    iters = len(train_loader)
    for idx, data in enumerate(train_loader):
        data = data.to(device)
        if verbose:
            print(data.y.shape, data.edge_index.shape)
        optimizer.zero_grad()
        output = model(data)
        if loss_fn is None:
            loss = F.cross_entropy(output, data.y.reshape(-1), weight=None, )
        else:
            loss = loss_fn(output, data.y.reshape(-1))

        if model.edge_weight is not None:
            l2_loss = 0
            if isinstance(model.edge_weight, nn.Module):  # nn.ParamterList
                for edge_weight in model.edge_weight:
                    l2_loss += 0.1 * t.mean((edge_weight) ** 2)
            elif isinstance(model.edge_weight, t.Tensor):
                l2_loss = 0.1 * t.mean((model.edge_weight) ** 2)
            loss += l2_loss
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        if not (scheduler is None):
            scheduler.step((epoch - 1) + idx / iters)
    return loss_all / iters


def test2(model, loader):
    model.eval()
    y_pred = []
    y_true = []
    y_output = []
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1].cpu().data.numpy()
        y = data.y.cpu().data.numpy()
        y_pred.extend(pred)
        y_true.extend(y)
        y_output.extend(output.cpu().data.numpy())

    acc = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    cuda_flag = args.cuda
    npz_file = args.infile
    save_folder = args.outdir
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag else 'cpu')

    os.makedirs(pathjoin(save_folder, 'models'), exist_ok=True)

    data = np.load(npz_file, allow_pickle=True)
    logExpr = data['logExpr'].T  # logExpr: row-cell, column-gene
    label = data['label']
    str_labels = data['str_labels']
    edge = t.tensor(data['edge_index'].T)
    edge = remove_self_loops(edge)[0]
    used_edge = edge.numpy()

    num_samples = logExpr.shape[0]

    init_lr = 0.01
    min_lr = 0.001
    max_epoch = 36
    batch_size = 64
    weight_decay = 1e-4
    dropout_ratio = 0.2
    extend_epoch = 30

    print('use wegithed cross entropy.... ')
    label_type = np.unique(label.reshape(-1))
    alpha = np.array([np.sum(label == x) for x in label_type])
    alpha = np.max(alpha) / alpha
    alpha = np.clip(alpha, 1, 50)
    alpha = alpha / np.sum(alpha)
    loss_fn = t.nn.CrossEntropyLoss(weight=t.tensor(alpha).float())
    loss_fn = loss_fn.to(device)

    dataset = ExprDataset(Expr=logExpr, edge=used_edge, y=label, device=device)
    gene_num = dataset.gene_num
    class_num = len(np.unique(label))
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False, collate_fn=collate_func,
                          drop_last=False)

    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for tr, ts in kf.split(X=label, y=label):
        train_index = tr
        test_index = ts

    train_dataset = dataset.split(t.tensor(train_index).long())
    test_dataset = dataset.split(t.tensor(test_index).long())
    # add more samples for those small celltypes
    train_dataset.duplicate_minor_types(dup_odds=50)

    num_workers = 0
    assert num_workers == 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                              collate_fn=collate_func, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_func)

    model = CA_SAGE(in_channel=dataset.num_expr_feaure, num_nodes=gene_num,
                    out_channel=class_num, edge_num=dataset.edge_num,
                    dropout_ratio=dropout_ratio,
                    ).to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay, )
    scheduler = CosineAnnealingWarmRestarts(optimizer, 2, 2, eta_min=min_lr, lr_max_decay=0.5)
    max_metric = float(0)
    weights_list = []

    for epoch in range(1, max_epoch):
        train_loss = train2(model, optimizer, train_loader, epoch, device, loss_fn, scheduler=scheduler)
        lr = optimizer.param_groups[0]['lr']
        print('epoch\t%03d,lr : %.06f,loss: %.06f' % (epoch, lr, train_loss))

    # train + valid
    print('stage 2 training...')
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for final_, valid_ in sss.split(dataset.y[train_index], dataset.y[train_index]):
        train_index2, valid_index2 = train_index[final_], train_index[valid_]
    valid_dataset = dataset.split(t.tensor(valid_index2).long())
    train_dataset = dataset.split(t.tensor(train_index2).long())
    if True:
        # add more samples for those small celltypes
        train_dataset.duplicate_minor_types(dup_odds=50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,collate_fn=collate_func, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,collate_fn=collate_func)
    lr = optimizer.param_groups[0]['lr']
    print('stage2 initilize lr:', lr)

    max_metric = float(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2, verbose=True,
                                                           min_lr=0.001)
    old_lr = lr
    for epoch_idx, epoch in enumerate(range(max_epoch, (max_epoch + extend_epoch))):
        train_loss = train2(model, optimizer, train_loader, epoch, device, loss_fn, verbose=False)
        valid_acc, valid_f1 = test2(model, valid_loader)

        lr = optimizer.param_groups[0]['lr']
        print('epoch\t%03d,lr : %.06f,loss: %.06f,' % (epoch,lr, train_loss,))
        scheduler.step(valid_f1)
        lr = optimizer.param_groups[0]['lr']
        if valid_f1 > max_metric:
            max_metric = valid_f1
            tmp_file = pathjoin(save_folder, 'models', 'bestmodel.pth')
            weights_list.append(tmp_file)
            t.save(model, tmp_file)

        if lr <= 0.001:
            break
    #get feature
    model = torch.load(save_folder+'/models'+'/bestmodel.pth').to(device)
    feature_init = np.ones((1, 1024))
    label = np.ones((1))
    for idx, data in enumerate(dataloader):
        output = getfeature(model, data)
        output = output.cpu().detach().numpy()
        feature = np.concatenate([feature_init, output], axis=0)
        numbers = data.y.detach().numpy()  # (1000,1)
        numbers = np.squeeze(numbers)
        numbers = [str_labels[x] for x in np.reshape(numbers.T, -1)]
        numbers = np.array(numbers, dtype=np.object)
        label = np.concatenate([label, numbers], axis=0)
    feature = np.delete(feature, [0, 0], axis=0)
    label = np.delete(label, 0)
    results_inner = {'Xtrain': feature.T, 'train_labels': label}
    sio.savemat(save_folder+'/datafeature.mat', {'train': results_inner})
    print('finish!')