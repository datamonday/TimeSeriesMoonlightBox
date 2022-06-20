import torch
import random
import numpy as np

from experiments.exp_main import Experiment
from utils.args_parser import get_arguments


# To ensure reproducibility
fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


if __name__ == '__main__':
    args = get_arguments()
    Exp = Experiment

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    iter_num = args.iter_num
    for iter in range(iter_num):
        setting = '{}_{}_{}_{}_sl-{}_pl-{}_ll-{}_n-layer-{}_hs-{}_iter-{}'.format(
            args.model_usage,
            args.model_use,
            args.filename,
            args.task,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.n_layers,
            args.hidden_size,
            # args.d_model,
            # args.n_heads,
            # args.enc_n_layers,
            # args.dec_n_layers,
            # args.d_ff,
            # args.factor,
            # args.embed,
            # args.distil,
            # args.des,
            iter
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

