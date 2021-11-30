import numpy as np
from Data.CWdata_dir import cw_0, cw_1, cw_2, cw_3
from data_generator.CWdata_generator import CW_preprocess
from torch.utils.data import DataLoader
from MTAGN_utils.train_utils import MyDataset


def CW_0():
    # 数据预处理参数
    Length = 2048  # 每个样本长度
    Number = 200  # 每类样本的数量
    Overlap = True  # 是否使用数据增强 CWdata Number超过60需要重叠取样
    Overlap_step = 512
    Shuffle = True
    Add_noise = False
    SNR = -5
    Normal = False  # 是否标准化
    Rate = (10, 20, 160)  # 训练集，验证集，测试集划分

    train_X = np.empty((0, Length), dtype=np.float32)
    train_Y1 = np.empty((1, 0), dtype=np.int64)
    train_Y2 = np.empty((1, 0), dtype=np.int64)
    valid_X = np.empty((0, Length), dtype=np.float32)
    valid_Y1 = np.empty((1, 0), dtype=np.int64)
    valid_Y2 = np.empty((1, 0), dtype=np.int64)
    test_X = np.empty((0, Length), dtype=np.float32)
    test_Y1 = np.empty((1, 0), dtype=np.int64)
    test_Y2 = np.empty((1, 0), dtype=np.int64)
    # NC
    trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(file_dir=cw_0[0],
                                                                                              data_len=Length,
                                                                                              sample_num=Number,
                                                                                              overlap=Overlap,
                                                                                              overlap_step=Overlap_step,
                                                                                              shuffle=Shuffle,
                                                                                              add_noise=Add_noise,
                                                                                              SNR=SNR,
                                                                                              normalize=Normal,
                                                                                              split_rate=Rate,
                                                                                              label1=0,
                                                                                              label2=0)
    train_X = np.vstack((train_X, trainX))
    train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
    valid_X = np.vstack((valid_X, validX))
    valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
    test_X = np.vstack((test_X, testX))
    test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)

    idx = 0
    for i in range(1, 4):
        for j in range(1, 4):
            idx += 1
            trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(
                file_dir=cw_0[idx],
                data_len=Length,
                sample_num=Number,
                overlap=Overlap,
                overlap_step=Overlap_step,
                shuffle=Shuffle,
                add_noise=Add_noise,
                SNR=SNR,
                normalize=Normal,
                split_rate=Rate,
                label1=i,
                label2=j)
            train_X = np.vstack((train_X, trainX))
            train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
            valid_X = np.vstack((valid_X, validX))
            valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
            test_X = np.vstack((test_X, testX))
            test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)
    train_X = np.reshape(train_X, (-1, 1, Length))
    valid_X = np.reshape(valid_X, (-1, 1, Length))
    test_X = np.reshape(test_X, (-1, 1, Length))

    return train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2


def CW_1():
    # 数据预处理参数
    Length = 2048  # 每个样本长度
    Number = 200  # 每类样本的数量
    Overlap = True  # 是否使用数据增强 CWdata Number超过60需要重叠取样
    Overlap_step = 512
    Shuffle = True
    Add_noise = False
    SNR = -5
    Normal = False  # 是否标准化
    Rate = (10, 20, 160)  # 训练集，验证集，测试集划分

    train_X = np.empty((0, Length), dtype=np.float32)
    train_Y1 = np.empty((1, 0), dtype=np.int64)
    train_Y2 = np.empty((1, 0), dtype=np.int64)
    valid_X = np.empty((0, Length), dtype=np.float32)
    valid_Y1 = np.empty((1, 0), dtype=np.int64)
    valid_Y2 = np.empty((1, 0), dtype=np.int64)
    test_X = np.empty((0, Length), dtype=np.float32)
    test_Y1 = np.empty((1, 0), dtype=np.int64)
    test_Y2 = np.empty((1, 0), dtype=np.int64)
    # NC
    trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(file_dir=cw_1[0],
                                                                                              data_len=Length,
                                                                                              sample_num=Number,
                                                                                              overlap=Overlap,
                                                                                              overlap_step=Overlap_step,
                                                                                              shuffle=Shuffle,
                                                                                              add_noise=Add_noise,
                                                                                              SNR=SNR,
                                                                                              normalize=Normal,
                                                                                              split_rate=Rate,
                                                                                              label1=0,
                                                                                              label2=0)
    train_X = np.vstack((train_X, trainX))
    train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
    valid_X = np.vstack((valid_X, validX))
    valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
    test_X = np.vstack((test_X, testX))
    test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)

    idx = 0
    for i in range(1, 4):
        for j in range(1, 4):
            idx += 1
            trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(
                file_dir=cw_1[idx],
                data_len=Length,
                sample_num=Number,
                overlap=Overlap,
                overlap_step=Overlap_step,
                shuffle=Shuffle,
                add_noise=Add_noise,
                SNR=SNR,
                normalize=Normal,
                split_rate=Rate,
                label1=i,
                label2=j)
            train_X = np.vstack((train_X, trainX))
            train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
            valid_X = np.vstack((valid_X, validX))
            valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
            test_X = np.vstack((test_X, testX))
            test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)
    train_X = np.reshape(train_X, (-1, 1, Length))
    valid_X = np.reshape(valid_X, (-1, 1, Length))
    test_X = np.reshape(test_X, (-1, 1, Length))

    return train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2


def CW_2():
    # 数据预处理参数
    Length = 2048  # 每个样本长度
    Number = 200  # 每类样本的数量
    Overlap = True  # 是否使用数据增强 CWdata Number超过60需要重叠取样
    Overlap_step = 512
    Shuffle = True
    Add_noise = False
    SNR = -5
    Normal = False  # 是否标准化
    Rate = (10, 20, 160)  # 训练集，验证集，测试集划分

    train_X = np.empty((0, Length), dtype=np.float32)
    train_Y1 = np.empty((1, 0), dtype=np.int64)
    train_Y2 = np.empty((1, 0), dtype=np.int64)
    valid_X = np.empty((0, Length), dtype=np.float32)
    valid_Y1 = np.empty((1, 0), dtype=np.int64)
    valid_Y2 = np.empty((1, 0), dtype=np.int64)
    test_X = np.empty((0, Length), dtype=np.float32)
    test_Y1 = np.empty((1, 0), dtype=np.int64)
    test_Y2 = np.empty((1, 0), dtype=np.int64)
    # NC
    trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(file_dir=cw_2[0],
                                                                                              data_len=Length,
                                                                                              sample_num=Number,
                                                                                              overlap=Overlap,
                                                                                              overlap_step=Overlap_step,
                                                                                              shuffle=Shuffle,
                                                                                              add_noise=Add_noise,
                                                                                              SNR=SNR,
                                                                                              normalize=Normal,
                                                                                              split_rate=Rate,
                                                                                              label1=0,
                                                                                              label2=0)
    train_X = np.vstack((train_X, trainX))
    train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
    valid_X = np.vstack((valid_X, validX))
    valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
    test_X = np.vstack((test_X, testX))
    test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)

    idx = 0
    for i in range(1, 4):
        for j in range(1, 4):
            idx += 1
            trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(
                file_dir=cw_2[idx],
                data_len=Length,
                sample_num=Number,
                overlap=Overlap,
                overlap_step=Overlap_step,
                shuffle=Shuffle,
                add_noise=Add_noise,
                SNR=SNR,
                normalize=Normal,
                split_rate=Rate,
                label1=i,
                label2=j)
            train_X = np.vstack((train_X, trainX))
            train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
            valid_X = np.vstack((valid_X, validX))
            valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
            test_X = np.vstack((test_X, testX))
            test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)
    train_X = np.reshape(train_X, (-1, 1, Length))
    valid_X = np.reshape(valid_X, (-1, 1, Length))
    test_X = np.reshape(test_X, (-1, 1, Length))

    return train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2


def CW_3():
    # 数据预处理参数
    Length = 2048  # 每个样本长度
    Number = 200  # 每类样本的数量
    Overlap = True  # 是否使用数据增强 CWdata Number超过60需要重叠取样
    Overlap_step = 512  # 重叠长度
    Shuffle = True
    Add_noise = False
    SNR = -5
    Normal = False  # 是否标准化
    Rate = (10, 20, 160)  # 训练集，验证集，测试集划分比例

    train_X = np.empty((0, Length), dtype=np.float32)
    train_Y1 = np.empty((1, 0), dtype=np.int64)
    train_Y2 = np.empty((1, 0), dtype=np.int64)
    valid_X = np.empty((0, Length), dtype=np.float32)
    valid_Y1 = np.empty((1, 0), dtype=np.int64)
    valid_Y2 = np.empty((1, 0), dtype=np.int64)
    test_X = np.empty((0, Length), dtype=np.float32)
    test_Y1 = np.empty((1, 0), dtype=np.int64)
    test_Y2 = np.empty((1, 0), dtype=np.int64)
    # NC
    trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(file_dir=cw_3[0],
                                                                                              data_len=Length,
                                                                                              sample_num=Number,
                                                                                              overlap=Overlap,
                                                                                              overlap_step=Overlap_step,
                                                                                              shuffle=Shuffle,
                                                                                              add_noise=Add_noise,
                                                                                              SNR=SNR,
                                                                                              normalize=Normal,
                                                                                              split_rate=Rate,
                                                                                              label1=0,
                                                                                              label2=0)
    train_X = np.vstack((train_X, trainX))
    train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
    valid_X = np.vstack((valid_X, validX))
    valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
    test_X = np.vstack((test_X, testX))
    test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)

    idx = 0
    for i in range(1, 4):
        for j in range(1, 4):
            idx += 1
            trainX, trainY1, trainY2, validX, validY1, validY2, testX, testY1, testY2 = CW_preprocess(
                file_dir=cw_3[idx],
                data_len=Length,
                sample_num=Number,
                overlap=Overlap,
                overlap_step=Overlap_step,
                shuffle=Shuffle,
                add_noise=Add_noise,
                SNR=SNR,
                normalize=Normal,
                split_rate=Rate,
                label1=i,
                label2=j)
            train_X = np.vstack((train_X, trainX))
            train_Y1, train_Y2 = np.append(train_Y1, trainY1), np.append(train_Y2, trainY2)
            valid_X = np.vstack((valid_X, validX))
            valid_Y1, valid_Y2 = np.append(valid_Y1, validY1), np.append(valid_Y2, validY2)
            test_X = np.vstack((test_X, testX))
            test_Y1, test_Y2 = np.append(test_Y1, testY1), np.append(test_Y2, testY2)
    train_X = np.reshape(train_X, (-1, 1, Length))
    valid_X = np.reshape(valid_X, (-1, 1, Length))
    test_X = np.reshape(test_X, (-1, 1, Length))

    return train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2


if __name__ == '__main__':
    train_x, train_y1, train_y2, valid_x, valid_y1, valid_y2, test_x, test_y1, test_y2 = CW_0()
    print(train_x.shape, train_y1.shape, train_y2.shape)
    print(type(train_x), type(train_y1), type(train_y2))
    print(valid_x.shape, valid_y1.shape, valid_y2.shape)
    print(test_x.shape, test_y1.shape, test_y2.shape)
    # print(test_y1)
    # print(test_y2)
    # for k in range(10):
    #     print('\n', train_y1[60 * k], train_y2[60 * k])

