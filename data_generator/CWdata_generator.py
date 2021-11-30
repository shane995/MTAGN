import os
import numpy as np
from scipy.io import loadmat
from Data.CWdata_dir import cw_0, cw_1, cw_2, cw_3
from MTAGN_utils.data_preprocess import add_wgn
from MTAGN_utils.data_preprocess import standardization


def data_labels_shuffle(data, label1, label2):
    """
    input 二维numpy array
    :param data: [data_num, data_len]
    :param label1: [data_num,]
    :param label2: [data_num,]
    :return:
    """
    np.random.seed(2021)
    index = np.arange(len(data))
    np.random.shuffle(index)
    return data[index], label1[index], label2[index]


def get_CW_data(file_dir):
    """
    input一个.mat文件，读取后是一个字典类型数据，需要获取其中“DE_time”对应key的value作为原始振动数据
    :param file_dir: 文件路径
    :return: 每个txt文件中的振动数据 [约122000,]的一维numpy array
    """
    temp_data = None
    file = loadmat(file_dir)
    for key in file.keys():
        if 'DE_time' in key:
            temp_data = file[key].flatten()
    temp_data = np.array(temp_data, dtype=np.float32)
    return temp_data


def CW_preprocess(file_dir, data_len=2048, sample_num=100, overlap=True, overlap_step=256, shuffle=False,
                  add_noise=False, SNR=5, normalize=True, split_rate=(60, 10, 30), label1=None, label2=None):
    """

    :param file_dir:
    :param data_len:
    :param sample_num:
    :param overlap:
    :param overlap_step:
    :param shuffle:
    :param add_noise:
    :param SNR:
    :param normalize:
    :param split_rate:
    :param label1:
    :param label2:
    :return:
    """
    def data_sampling(raw_data):
        """
        对原始数据(122000个点)进行取样
        :param raw_data: [122000,]
        :return: samples: [num, data_len]
        """
        samples = np.empty((0, data_len), dtype=np.float32)
        if not overlap:  # 不重叠取样
            start = 0
            for i in range(sample_num):
                sample = raw_data[start: start + data_len]
                samples = np.vstack((samples, sample))
                start += data_len
        else:  # 重叠取样
            start = 0
            for i in range(sample_num):
                sample = raw_data[start: start + data_len]
                samples = np.vstack((samples, sample))
                start += overlap_step
        if add_noise:
            samples = add_wgn(samples, snr=SNR)
        if normalize:
            samples = standardization(samples)
        return samples

    def add_labels(x):
        """
        add fault type label and fault severity label
        :param x:
        :return: labels1-[data_len,], labels2-[data_len,]
        """
        len_x = len(x)
        labels_1 = np.ones((len_x,), dtype=np.int64) * label1
        labels_2 = np.ones((len_x,), dtype=np.int64) * label2
        return labels_1, labels_2

    def split_data(samples, labels1, labels2):
        """
        split data into train_set, valid_set and test_set
        :param samples:
        :param labels1:
        :param labels2:
        :return:
        """
        # if sum(split_rate) != sample_num:
        #     print("error, sum(split_rate != sample_num)")
        #     exit()
        end_point1 = int(split_rate[0])
        end_point2 = int(split_rate[0] + split_rate[1])
        end_point3 = sum(split_rate)
        x_train, y1_train, y2_train = samples[:end_point1], labels1[:end_point1], labels2[:end_point1]
        x_valid, y1_valid, y2_valid = samples[end_point1: end_point2], labels1[end_point1: end_point2], \
                                      labels2[end_point1: end_point2]
        x_test, y1_test, y2_test = samples[end_point2: end_point3], labels1[end_point2: end_point3], \
                                   labels2[end_point2: end_point3]
        return x_train, y1_train, y2_train, x_valid, y1_valid, y2_valid, x_test, y1_test, y2_test

    data = get_CW_data(file_dir)
    Samples = data_sampling(data)
    Labels1, Labels2 = add_labels(Samples)
    if shuffle:
        Samples, Labels1, Labels2 = data_labels_shuffle(Samples, Labels1, Labels2)
    Train_x, Train_y1, Train_y2, Valid_x, Valid_y1, Valid_y2, Test_x, Test_y1, Test_y2 = split_data(Samples,
                                                                                                    Labels1,
                                                                                                    Labels2)
    return Train_x, Train_y1, Train_y2, Valid_x, Valid_y1, Valid_y2, Test_x, Test_y1, Test_y2


if __name__ == '__main__':
    file_path = cw_0[0]
    train_x, train_y1, train_y2, valid_x, valid_y1, valid_y2, \
    test_x, test_y1, test_y2 = CW_preprocess(file_path, sample_num=200, split_rate=(60, 40, 80), label1=0, label2=0)
    print(train_x.shape, train_y1.shape, train_y2.shape)
    print(valid_x.shape, valid_y1.shape, valid_y2.shape)
    print(test_x.shape, test_y1.shape, test_y2.shape)

