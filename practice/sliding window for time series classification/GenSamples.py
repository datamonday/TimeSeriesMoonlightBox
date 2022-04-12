import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical


class GenerateSamples():
    '''
    @author：datamonday
    
    功能说明：加载处理好的数据，并通过滑动窗口实现划分样本
    '''

    def __init__(self, train_file_list, test_file_list, rootdir,
                 sw_width, sw_steps):
        '''
        初始化参数
        ----------
        参数说明：
            train_file_list：所有拼接好的训练数据文件名列表；
            test_conc_files：所有拼接好的测试数据文件名列表；
        '''
        self.train_file_list = train_file_list
        self.test_file_list = test_file_list

        self.train_loaded = []
        self.test_loaded = []

        # super(GenerateSamples, self).__init__(rootdir, filename_len, true_label, drop_col, new_col, sw_width, sw_steps, sheet_name=0) # 重写父类中的rootdir属性
        self.rootdir = rootdir
        self.sw_steps = sw_steps
        self.sw_width = sw_width

        self.new_columns = []

    def slide_window(self, rows):
        '''
        函数功能：
        生成切片列表截取数据，按指定窗口宽度的50%重叠生成；
        --------------------------------------------------
        参数说明：
        rows：excel文件中的行数；
        size：窗口宽度；
        '''

        start = 0
        s_num = (rows - self.sw_width) // self.sw_steps  # 计算滑动次数
        new_rows = self.sw_width + (self.sw_steps * s_num)  # 为保证窗口数据完整，丢弃不足窗口宽度的采样数据

        while True:
            if (start + self.sw_width) > new_rows:  # 丢弃不是完整窗口的数据
                return
            yield start, start + self.sw_width
            start += self.sw_steps

    def segment_sensor_signal(self, data_file, label_index='state'):
        '''
        参数说明：
        self.sw_width：滑动窗口宽度；
        n_features：特征数量；
        label_index：用于计数的列索引；添加编码后的标签列

        '''
        # 计算特征数
        n_features = data_file.shape[1] - 1
        # 添加编码后的列名
        label_index = 'state_encode'
        # scikit-learn 类，实现将真实标签转化为整型数字；
        le = preprocessing.LabelEncoder()
        # 添加新的标签列；ravel 返回包含输入元素的一维数组。
        data_file[label_index] = le.fit_transform(data_file['state'].values.ravel())
        print(data_file.columns[:-2])

        # -----------------------------------------------#
        # label_index = 'state'
        # 构造一个切片，方便填充数据
        segments = np.empty((0, self.sw_width, n_features), dtype=np.float64)
        # 单列标签数据
        # labels = np.empty((0), dtype=np.float64)
        labels = []
        labels_true = np.empty((0))

        for start, end in self.slide_window(data_file.shape[0]):  # 调用滑动窗口函数，通过yield实现滑动效果；
            temporary = []  # 每次存放各个特征的序列片段
            for feature in data_file.columns[:-2]:  # 遍历文件所有特征列
                temporary.append(data_file[feature][start:end])

            if (len(data_file[label_index][start:end]) == self.sw_width):  # 如果达到窗口宽度，则截取样本
                # 将数据通过stack方法堆叠成样本 shape为（none, sw_width, features）；
                segments = np.vstack([segments, np.dstack(temporary)])  # 堆叠为三维数组

                # scipy.stats.mode函数寻找数组每行/每列中最常出现成员以及出现的次数；实现将一个窗口内采样数据的标签出现次数最多的作为样本标签
                labels = np.append(labels, stats.mode(data_file[label_index][start:end])[0][0])  # 出现次数最多的标签作为样本标签
                labels_true = np.append(labels_true, stats.mode(data_file['state'][start:end])[0][0])

        # labels_true = pd.DataFrame(labels)
        # labels_onehot = np.asarray(pd.get_dummies(labels), dtype = np.int8)

        labels = np.asarray(labels, dtype=np.float64)
        labels_onehot = to_categorical(labels, num_classes=len(np.unique(data_file[label_index].values)))
        return segments, labels_onehot, labels_true

    def load_conc_files(self, filename):
        '''
        加载单个拼接好的某个工况的csv文件；
        '''
        data = pd.read_csv(self.rootdir + '/' + filename, header=0)
        self.new_columns = data.columns
        return data

    def load_all_file(self):
        '''
        加载所有文件
        '''
        for file in self.train_file_list:
            self.train_loaded.append(self.load_conc_files(file))

        for file in self.test_file_list:
            self.test_loaded.append(self.load_conc_files(file))

        self.train_loaded = np.vstack(self.train_loaded)  # 堆叠，沿第二维堆叠（二维数组沿列堆叠）
        print(f'train_loaded.shape:{self.train_loaded.shape}\n')  # 打印堆叠后的shape
        self.train_loaded = pd.DataFrame(self.train_loaded, columns=self.new_columns)  # 转换为DataFrame格式

        self.test_loaded = np.vstack(self.test_loaded)  # 堆叠，沿第二维堆叠（二维数组沿列堆叠）
        print(f'test_loaded.shape:{self.test_loaded.shape}\n')  # 打印堆叠后的shape
        self.test_loaded = pd.DataFrame(self.test_loaded, columns=self.new_columns)  # 转换为DataFrame格式

    def normalization(self, data):
        '''
        Z-Socre 标准化
        '''
        label = data['state']
        data.drop(columns='state', inplace=True)
        data = (data - data.mean()) / data.std()
        data['state'] = label
        return data

    def gen_train_test_samples(self, norm=True):
        '''
        生成样本
        '''
        if norm == True:
            self.train_loaded = self.normalization(self.train_loaded)
            self.test_loaded = self.normalization(self.test_loaded)

        trainX, trainy, train_true_label = self.segment_sensor_signal(self.train_loaded)
        testX, testy, test_true_label = self.segment_sensor_signal(self.test_loaded)

        print('-------Processing Done!------')

        print('trainX.shape:{}\ntrainy.shape:{}\ntestX.shape:{}\ntesty.shape:{}\n'.format(trainX.shape, trainy.shape,
                                                                                          testX.shape, testy.shape))

        return trainX, trainy, testX, testy, train_true_label, test_true_label

# if __name__ == '__main__':
#
#     rootdir = 'D:/GraduationCode/01 Datasets/Carsim/Processed dataset 9 features/'
#     #     train_file_list = ['train_normal.csv', 'train_turn left.csv', 'train_turn right.csv', 'train_accelerate.csv', 'train_decelerate.csv',
#     #                    'train_lane change.csv', 'train_emergency brake.csv', 'train_side slip.csv', 'train_dp side slip.csv']
#
#     #     test_file_list = ['test_normal.csv', 'test_turn left.csv', 'test_turn right.csv', 'test_accelerate.csv', 'test_decelerate.csv',
#     #                        'test_lane change.csv', 'test_emergency brake.csv', 'test_side slip.csv', 'test_dp side slip.csv']
#     train_file_list = ['train_10class_9features.csv']
#     test_file_list = ['test_10class_9features.csv']
#
#     sw_width = 24  # 滑动窗口的宽度
#     sw_steps = 12  # 滑动步长
#
#     gensamples = GenerateSamples(train_file_list, test_file_list, rootdir,
#                                  sw_width, sw_steps)
#     gensamples.load_all_file()
#
#     trainX, trainy, testX, testy, train_true_label, test_true_label = gensamples.gen_train_test_samples(norm=False)