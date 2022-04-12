import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys


# 实现多变量的滑动窗口截取数据(滑动步长为1)
def create_samples(self, dataset, target, start_index, end_index):
    '''
    参数说明：
    dataset：DataFrame.values 格式；
    target：要预测的特征列，格式同上；
    start_index：数据中开始截取的索引；
    end_index：数据中结束截取的索引；
    '''
    # 定义用于存放样本数据的列表
    sample = []
    # 定义用于存放样本期望预测值的列表
    labels = []

    # 令开始索引为start_index参数+预测的时间步长度，后续程序段会将其减去
    # 该函数返回样本之前的预测时间步长度，目的是为了防止数据开始0之前没有数据而无法截取。
    start_index = start_index + self.n_history

    if end_index is None:
        # 结束截取索引为数据集行数减去预测的时间步长度；
        # 此举是为了保证截取的样本完整，如果最后一个样本不满足预测的时间步长度则丢弃。
        end_index = len(dataset) - self.n_outputs

    for i in range(start_index, end_index):
        # 防止最后一个样本溢出
        if i + self.n_outputs >= end_index:
            pass

        else:
            # 按照步长返回n_history长度大小的切片
            indices = range(i - self.n_history, i, self.interval) # step表示滑动步长

            # 截取数据并添加到列表中
            sample.append(dataset[indices])

            # 如果单步预测标志位为True，则期望预测值为之后的一个单值
            if self.single_step:
                labels.append(target[i+ self.n_outputs])

            # 否则为设置的目标长度值
            else:
                labels.append(target[i:i+ self.n_outputs])

    sample = np.array(sample)
    labels = np.array(labels)

    return sample, labels


# 实现多变量的滑动窗口截取数据(滑动步长为step)
def create_samples2(dataset, target, start_index, end_index, history_size,
                    target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step) # step表示滑动步长
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)