import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='Ped.csv', scale=True, timeenc=0, freq='h'):
        # size[seq_len, label_len, pred_len]
        # info
        super(Dataset_Custom, self).__init__()

        if size == None:
            raise Exception("there is no seq_len")
        else:
            self.seq_len = size[0]  # 输入长度
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 指定目前是啥操作

        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        # cols = list(df_raw.columns)
        cols_data = df_raw.columns[1:]

        num_train = int(len(df_raw) * 0.7)  #
        num_test = int(len(df_raw) * 0.2)  #
        num_val = len(df_raw) - num_train - num_test  #

        start_index = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        end_index = [num_train, num_train + num_val, len(df_raw)]
        border_s = start_index[self.set_type]
        border_e = end_index[self.set_type]

        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[start_index[0]:end_index[0]]
            self.scaler.fit(train_data.values)  # 对每列求均值 和 标准差
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border_s:border_e]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)  # 处理时间特征

        self.data_x = data[border_s:border_e]
        self.data_y = data[border_s:border_e]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        pre_begin = s_end - self.label_len
        pre_end = pre_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[pre_begin:pre_end]

        seq_x_time = self.data_stamp[s_begin:s_end]
        seq_y_time = self.data_stamp[pre_begin:pre_end]

        return seq_x, seq_y, seq_x_time, seq_y_time

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path,flag='pred', size=None,
                 data_path='Ped.csv', scale=True, timeenc=0, freq='h'):
        # info
        assert flag=='pred'
        super( Dataset_Pred, self).__init__()

        if size == None:
            raise Exception("there is no seq_len")
        else:
            self.seq_len = size[0]  # 输入长度
            self.label_len = size[1]
            self.pred_len = size[2]

        self.set_type = 2
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols_data = df_raw.columns[1:]

        num_train = int(len(df_raw) * 0.7)  #
        num_test = int(len(df_raw) * 0.1)  #
        num_val = len(df_raw) - num_train - num_test  #

        start_index = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        end_index = [num_train, num_train + num_val, len(df_raw)]
        border_s = start_index[self.set_type]
        border_e = end_index[self.set_type]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[start_index[0]:end_index[0]]
            self.scaler.fit(train_data.values)  # 对每列求均值 和 标准差
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border_s:border_e]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)  # 处理时间特征

        self.data_x = data[border_s:border_e]
        self.data_y = data[border_s:border_e]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        pre_begin = s_end - self.label_len
        pre_end = pre_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[pre_begin:pre_end]

        seq_x_time = self.data_stamp[s_begin:s_end]
        seq_y_time = self.data_stamp[pre_begin:pre_end]

        return seq_x, seq_y, seq_x_time, seq_y_time

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
