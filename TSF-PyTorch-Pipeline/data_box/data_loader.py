import os

import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # Calculate the mean value and the standard deviation by column.
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        """
        z = (x - u) / s
        """
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class Data2Sample(Dataset):
    def __init__(self,
                 root_dir,
                 filename,
                 seq_len,
                 pred_len,
                 split_ratios,
                 n_feats=14,
                 label_len=0,
                 which_set='train',
                 task='MISO',  # multi-input-single-output task
                 freq='10min',
                 target='T (degC)',
                 timestamp_name='Date Time',
                 use_cols=None,
                 is_scale=True,  # MinMaxScaler, StandardScaler
                 time_enc_method=0,  # time feature encode method
                 use_gpu=True,
                 ):
        self.root_dir = root_dir
        self.filename = filename

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        assert seq_len, "You must provide seq_len!"
        assert pred_len, "You must provide pred_len!"

        assert which_set in ['train', 'val', 'test']
        self.which_set = which_set
        self.split_ratios = split_ratios

        # This use to choose dataset borders
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[which_set]

        self.task = task
        self.freq = freq
        self.target = target
        self.timestamp_name = timestamp_name
        self.is_scale = is_scale
        self.time_enc_method = time_enc_method

        self.use_cols = use_cols
        self.use_gpu = use_gpu
        self.n_feats = n_feats

        self.scale = np.ones(self.n_feats)
        self.scale = torch.from_numpy(self.scale).float()
        if self.use_gpu:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_dir, self.filename))

        print('number of df_raw.columns: ', len(df_raw.columns))

        if self.freq == 'hour':
            self.grain_size = 1
        elif self.freq == '5min':
            self.grain_size = 60 // 5
        elif self.freq == '10min':
            self.grain_size = 60 // 10
        elif self.freq == '15min':
            self.grain_size = 60 // 15
        else:
            raise ValueError('Unknown freq parameter, you only can choose from ["hour", "5min", "10min", "15min"]!')

        if self.use_cols:
            cols = self.use_cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove(self.timestamp_name)

        # This ensures that the first column is the 'date' and the last column is the 'target'.
        # look like df_raw.columns = ['date', ...(other features), target feature].
        df_raw = df_raw[[self.timestamp_name] + cols + [self.target]]

        df_raw_rows = df_raw.shape[0]

        if self.split_ratios:
            tr_ratio, va_ratio, te_ratio = self.split_ratios
        else:
            tr_ratio, va_ratio, te_ratio = [0.7, 0.2, 0.1]
        tr_size = int(df_raw_rows * tr_ratio)
        va_size = int(df_raw_rows * va_ratio)
        te_size = df_raw_rows - tr_size - va_size

        # Start and end indices for intercepting train, test and validation sets
        # train set: [0, tr_size]
        # valid set: [tr_size - seq_len, tr_size + va_size]
        # test  set: [df_raw_rows - te_size - seq_len, df_raw_rows]
        border_start_list = [0, tr_size - self.seq_len, df_raw_rows - te_size - self.seq_len]
        border_end_list = [tr_size, tr_size + va_size, df_raw_rows]

        border_start, border_end = border_start_list[self.set_type], border_end_list[self.set_type]

        # Generate uni-variate or multivariate series.
        if self.task == 'MISO' or self.task == 'MIMO':
            # Use features other than timestamp columns
            df_data = df_raw[df_raw.columns[1:]]
        elif self.task == 'SISO':
            df_data = df_raw[[self.target]]
        else:
            raise ValueError('Unknown task method! You must choose from ["MISO", "MIMO", "SISO"].')

        # Normalize dataset use train set.
        # Choose the normalize method
        if self.is_scale:
            self.scaler = StandardScaler()
            train_border_start, train_border_end = border_start_list[0], border_end_list[0]
            train_data = df_data[train_border_start:train_border_end]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # X, y
        self.data_x = data[border_start:border_end]
        self.data_y = data[border_start:border_end]

    def __getitem__(self, index):
        # start and end index of a single sample data_boxes.
        s_begin = index
        s_end = s_begin + self.seq_len

        # start and end index of a single sample label. The label_len param can be set zero.
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # shape: two dim without batch size, [seq_len, n_feats]
        seq_x = self.data_x[s_begin:s_end]
        #  shape: two dim without batch size, [pred_len, n_feats]
        seq_y = self.data_y[r_begin:r_end]
        # print(f'seq_x.shape: {seq_x.shape}, seq_y.shape: {seq_y.shape}.')

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)