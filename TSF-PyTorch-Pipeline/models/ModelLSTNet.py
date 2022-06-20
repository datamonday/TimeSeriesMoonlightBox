import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    # def __init__(self,
    #              seq_len: int,
    #              n_feats: int,
    #              n_rnn_hid: int=72,
    #              n_cnn_hid: int=72,
    #              n_skip_hid: int=5,
    #              cnn_kernel: int=6,
    #              skip: int=25,
    #              highway_len: int=24,
    #              dropout_prob: float=0.2,
    #              output_func: str = 'sigmoid',
    #              ):
    def __init__(self,
                 args,
                 n_rnn_hid: int = 72,
                 n_cnn_hid: int = 72,
                 n_skip_hid: int = 5,
                 cnn_kernel: int = 6,
                 skip: int = 25,
                 highway_len: int = 24,
                 dropout_prob: float = 0.2,
                 output_func: str = 'sigmoid',
                 ):
        super().__init__()

        self.seq_len = args.seq_len
        self.n_feats = args.n_feats

        self.n_rnn_hid = n_rnn_hid
        self.n_cnn_hid = n_cnn_hid
        self.n_skip_hid = n_skip_hid
        self.cnn_kernel = cnn_kernel

        self.skip = skip
        self.highway_len = highway_len

        # The parameter pt is the number of hidden cells to be skipped.
        # The value of pt can be easily determined and must be adjusted.
        self.pt = int((self.seq_len - self.cnn_kernel) / self.skip)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.n_cnn_hid, kernel_size=(self.cnn_kernel, self.n_feats))
        self.gru1 = nn.GRU(input_size=self.n_cnn_hid, hidden_size=self.n_rnn_hid, num_layers=1)
        self.dropout = nn.Dropout(p=dropout_prob)

        # recurrent skip model part
        if self.skip > 0:
            self.skip_gru = nn.GRU(input_size=self.n_cnn_hid, hidden_size=self.n_skip_hid, num_layers=1)
            self.linear1 = nn.Linear(in_features=self.n_rnn_hid + self.skip * self.n_skip_hid, out_features=self.n_feats)
        else:
            self.linear1 = nn.Linear(in_features=self.n_rnn_hid, out_features=self.n_feats)

        # autoregressive linear model part
        if self.highway_len > 0:
            self.highway = nn.Linear(self.highway_len, 1)

        self.output = None
        if output_func == 'sigmoid':
            self.output_func = torch.sigmoid
        else:
            self.output_func = torch.tanh

    def forward(self, y_c):
        batch_size = y_c.size(0)

        # CNN
        conv_in = y_c.view(-1, 1, self.seq_len, self.n_feats)
        # [bcz, 1, seq_len, n_feats]
        conv_ot = self.conv1(conv_in)
        conv_ot = self.dropout(F.relu(conv_ot))
        # []
        # squeeze(): Returns a tensor with all the dimensions of input of size 1 removed.
        conv_ot = torch.squeeze(conv_ot, 3)

        # RNN
        rnn_in = conv_ot.permute(2, 0, 1).contiguous()
        output, final_h = self.gru1(rnn_in)
        rnn_ot = self.dropout(torch.squeeze(final_h, 0))

        # Skip-RNN
        if self.skip > 0:
            skip_in = conv_ot[:, :, int(-self.pt * self.skip):].contiguous()
            skip_in = skip_in.view(batch_size, self.n_rnn_hid, self.pt, self.skip)
            skip_in = skip_in.permute(2, 0, 3, 1).contiguous()
            skip_in = skip_in.view(self.pt, batch_size * self.skip, self.n_rnn_hid)

            output, final_h = self.skip_gru(skip_in)
            final_h = final_h.view(batch_size, self.skip * self.n_skip_hid)
            skip_rnn_ot = self.dropout(final_h)

            rnn_ot = torch.cat((rnn_ot, skip_rnn_ot), 1)

        res = self.linear1(rnn_ot)

        # highway
        if self.highway_len > 0:
            hw_in = y_c[:, -self.highway_len:, :]
            hw_in = hw_in.permute(0, 2, 1).contiguous().view(-1, self.highway_len)
            hw_ot = self.highway(hw_in)
            hw_ot = hw_ot.view(-1, self.n_feats)
            res = res + hw_ot

        if self.output:
            res = self.dropout(res)

        return res

