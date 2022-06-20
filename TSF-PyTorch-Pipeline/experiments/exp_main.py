import numpy as np
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.exp_basic import ExpBasic
from data_box.data_loader import Data2Sample
from models import ModelLSTM, ModelGRU, ModelLSTNet
from utils.metrics import metric
from utils.train_utils import EarlyStopping, adjust_learning_rate
from utils.plot_utils import visual


class Experiment(ExpBasic):
    def __init__(self, args):
        super().__init__(args)

        self.args = args

    def get_model(self):
        model_map = {
            'LSTM': ModelLSTM,
            'GRU': ModelGRU,
            'LSTNet': ModelLSTNet,
        }

        self.model = model_map[self.args.model_use].Model(self.args).float()
        return self.model

    def get_args(self):
        return self.args

    def get_dataset(self, which_set):
        if which_set == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        data_set = Data2Sample(
            root_dir=self.args.root_dir,
            filename=self.args.filename,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            split_ratios=self.args.split_ratios,
            n_feats=self.args.n_feats,
            label_len=self.args.label_len,
            which_set=which_set,
            task=self.args.task,  # multi-input-single-output task
            freq=self.args.freq,
            target=self.args.target,
            timestamp_name=self.args.timestamp_name,
            use_cols=self.args.use_cols,
            is_scale=self.args.is_scale,  # MinMaxScaler, StandardScaler
            time_enc_method=self.args.time_enc_method,  # time feature encode method
            use_gpu=self.args.use_gpu,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last,
            num_workers=self.args.num_workers,
        )

        return data_set, data_loader

    def get_optimizer(self):
        if self.args.optimizer == 'sgd':
            # Stochastic Gradient Optimization, 1964. use current gradients (aL/aW) * learning_rate (lr) to update gradients.
            model_optim = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adagrad':
            # Adaptive Gradient, 2011.
            model_optim = torch.optim.Adagrad(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'rmsprop':
            # Root Mean Square Prop, 2012.
            model_optim = torch.optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adadelta':
            # Adaptive Delta, similar to rmsprop, 2012.
            model_optim = torch.optim.Adadelta(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamax':
            # Adaptive Max, 2015. The Adam authors use infinity norms (hence the name "max") as an improvement to the Adam optimizer.
            model_optim = torch.optim.Adamax(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'nadam':
            # Nesterov Adam, 2015.
            model_optim = torch.optim.NAdam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            # Decoupled Weight Decay Regularization, 2017.
            model_optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'amsgrad':
            # On the Convergence of Adam and Beyond, 2018.
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, amsgrad=True)
        else:
            # Adaptive Moment Estimation, 2014.
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, amsgrad=False)

        return model_optim

    def get_criterion(self):
        if self.args.loss_function == 'mae':
            # measures the mean absolute error (MAE) between each element in the input x and target y.
            criterion = nn.L1Loss()
        elif self.args.loss_function == 'smoothl1loss':
            # uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise.
            # It is less sensitive to outliers than torch.nn.MSELoss and in some cases prevents exploding gradients
            criterion = nn.SmoothL1Loss()
        elif self.args.loss_function == 'huberloss':
            # uses a squared term if the absolute element-wise error falls below delta and a delta-scaled L1 term otherwise.
            # This loss combines advantages of both L1Loss and MSELoss;
            # the delta-scaled L1 region makes the loss less sensitive to outliers than MSELoss,
            # while the L2 region provides smoothness over L1Loss near 0.
            criterion = nn.HuberLoss()
        else:
            #  measures the mean squared error (squared L2 norm) between each element in the input x and target y.
            criterion = nn.MSELoss()
        return criterion

    def process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        # model prediction values
        prediction = self.model(batch_x)

        # which dimension(variable) to forecast
        f_dim = -1 if self.args.task == 'MISO' else 0

        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].float().to(self.device)
        prediction = prediction[..., :, f_dim:].float()

        return prediction, batch_y

    def vali(self, data_set, data_loader, criterion):
        validation_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(data_loader):
                batch_pred, batch_truth = self.process_one_batch(batch_x, batch_y)
                loss = criterion(batch_pred, batch_truth)
                validation_loss.append(loss.item())
        validation_loss = np.average(validation_loss)
        return validation_loss

    def train(self, setting):
        train_data, train_loader = self.get_dataset(which_set='train')
        vali_data, vali_loader = self.get_dataset(which_set='val')
        test_data, test_loader = self.get_dataset(which_set='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self.get_optimizer()
        criterion = self.get_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_pred, batch_truth = self.process_one_batch(batch_x, batch_y)

                loss = criterion(batch_pred, batch_truth)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self.args.verbose > 0:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

    def test(self, setting, use_best_model=0):
        test_data, test_loader = self.get_dataset(which_set='test')
        if use_best_model:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        folder_path = './test_resutls/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):

                outputs, batch_y = self.process_one_batch(batch_x, batch_y)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                trues.append(true)
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

            return

