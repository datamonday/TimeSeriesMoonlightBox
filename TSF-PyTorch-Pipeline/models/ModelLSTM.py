import os

import torch
from torch import nn


class Model(nn.Module):
    # def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.n_feats
        self.hidden_size = args.hidden_size
        self.num_layers = args.n_layers
        self.output_size = args.pred_len

        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu_id) if not self.args.use_multi_gpu else self.args.devices
            self.device = torch.device('cuda:{}'.format(self.args.gpu_id))
            print('Use GPU: cuda:{}'.format(self.args.gpu_id))
        else:
            self.device = torch.device('cpu')
            print('Use CPU')

        self.out_var = self.args.out_var

        self.num_directions = 1

        self.batch_size = args.batch_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # input shape: (batch_size, seq_len, input_size)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)

        # output shape: (batch_size, seq_len, num_directions * hidden_size)
        output, hidden = self.lstm(input_seq, (h_0, c_0))

        # pred shape: (batch_size, seq_len, output_size)
        pred = self.linear(output)
        # pred = pred[:, -1, :]
        pred = pred[:, -self.output_size:, -self.out_var:]

        return pred

