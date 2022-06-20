import os
import torch


class ExpBasic(object):
    def __init__(self, args):
        self.args = args
        self.device = self.acquire_device()
        self.model = self.get_model().to(self.device)

    def get_model(self):
        raise NotImplementedError
        return None

    def acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_id) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu_id))
            print('Use GPU: cuda:{}'.format(self.args.gpu_id))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def get_dataset(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass