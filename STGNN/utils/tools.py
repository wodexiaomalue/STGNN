import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class causal_conv(nn.Module):
    def __init__(self, kernelsize, padding, in_channel, out_channel):
        super(causal_conv, self).__init__()
        self.padding = padding
        self.conv =nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernelsize, padding=padding)

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1))[:, :, :-self.padding].permute(0, 2, 1)


class multi_locality(nn.Module):
    def __init__(self, kernelsize, in_channel, out_channel):
        super(multi_locality, self).__init__()

        self.query_conv = nn.ModuleList([causal_conv(size, size-1, in_channel, out_channel) for size in kernelsize])
        self.key_conv = nn.ModuleList([causal_conv(size, size-1, in_channel, out_channel) for size in kernelsize])
        self.layer = nn.Linear(1, len(kernelsize))

    def forward(self, queries, keys):
        query_list = []
        key_lsit = []
        weight = nn.Softmax(-1)(self.layer(queries.unsqueeze(-1)))
        for q_conv in self.query_conv:
            query = q_conv(queries)
            query_list.append(query.unsqueeze(-1))
        for k_conv in self.key_conv:
            key = k_conv(keys)
            key_lsit.append(key.unsqueeze(-1))
        query = torch.cat(query_list, dim=-1)
        key = torch.cat(key_lsit, dim=-1)

        return torch.sum(query*weight, dim=-1), torch.sum(key*weight, dim=-1)


def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

def load_norm_Laplacian(adj_path, device, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj_np = np.array(adj_df, dtype=dtype)
    assert adj_np.shape[0] == adj_np.shape[1]
    N = adj_np.shape[0]
    adj_np = adj_np + np.identity(N)
    D = np.diag(np.sum(adj_np, axis=1))
    sym_norm_Adj_matrix = np.dot(np.sqrt(D), adj_np)
    sym_norm_Adj_matrix = np.dot(sym_norm_Adj_matrix, np.sqrt(D))
    adj = torch.from_numpy(sym_norm_Adj_matrix).to(device)
    return adj, N


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            _lr = param_group['lr']
            param_group['lr'] = lr
        print('Updating learning rate {0} to {1}'.format(_lr, lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # for v in model.parameters():
        #     1
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
