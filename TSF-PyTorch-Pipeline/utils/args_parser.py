import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

    # basic configuration
    parser.add_argument('--iter_num', type=int, default=5, help='experiments times')

    # data preprocessing
    parser.add_argument('--root_dir', type=str, default='D:\\XDUCampus\\HW-03-TSForecasting\\', required=True)
    parser.add_argument('--filename', type=str, default='processed_data.csv', required=True)
    # ['p (mbar)', 'VPmax (mbar)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'Wx', 'Wy', 'max Wx', 'max Wy', 'T (degC)']
    parser.add_argument('--n_feats', type=int, default=9, help='The number of features for training model.')

    parser.add_argument('--seq_len', type=int, default=24 * 6, )
    parser.add_argument('--pred_len', type=int, default=1 * 6, )
    parser.add_argument('--split_ratios', type=list, default=[0.7, 0.2, 0.1], )
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--task', type=str, default='MISO', help='forecasting task, options:[MIMO, SISO, MISO]')
    parser.add_argument('--freq', type=str, default='10min')
    parser.add_argument('--target', type=str, default='T (degC)')
    parser.add_argument('--out_var', type=int, default=1)
    parser.add_argument('--timestamp_name', type=str, default='Date Time')
    parser.add_argument('--use_cols', type=list, default=None)
    parser.add_argument('--is_scale', type=bool, default=True)
    parser.add_argument('--time_enc_method', type=int, default=0)

    # training configuration
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # model
    parser.add_argument('--model_usage', type=str, required=True, default='train', help='model id')
    parser.add_argument('--model_use', type=str, required=True, default='GRU', help='model name, options: [GRU, LSTM, LSTNet]')
    parser.add_argument('--hidden_size', type=int, default=48)
    parser.add_argument('--n_layers', type=int, default=2)

    # model training parameters
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--loss_function', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--verbose', type=int, default=1, help='whether to print metrics in training.')

    return parser.parse_args()





