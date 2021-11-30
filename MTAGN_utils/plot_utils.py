import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.fft import fft, fftfreq


def plot_confusion_matrix(y_ture, y_pred, norm=True, task=1, fig_name='test'):
    """

    :param task:
    :param y_ture: (n,)
    :param y_pred: (n,)
    :param norm: 是否归一化
    :param fig_name: 图片名
    :return:
    """
    # plt.rc对全图字体进行统一修改:
    font = {'family': 'Times New Roman',
            'weight': 'light',
            'size': 12,
            }
    plt.rc('font', **font)
    # plt.rc('font', family='Times New Roman', style='normal', weight='light', size='1')
    f, ax = plt.subplots()
    cm = confusion_matrix(y_ture, y_pred)
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 12,
             }

    if norm:  # # 归一化,可显示准确率,默认显示
        cm = cm.astype('float32') / (cm.sum(axis=1)[:, np.newaxis])
        if task == 1:
            sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='.2f',
                        linewidths=0.02, linecolor="w", vmin=0, vmax=1)
            ax.set_xlabel('Predicted label ($T_1$)', fontdict=font1)
            ax.set_ylabel('True label ($T_1$)', fontdict=font1)
        elif task == 2:
            sns.heatmap(cm, annot=True, ax=ax, cmap='Greens', fmt='.2f',
                        linewidths=0.02, linecolor="w", vmin=0, vmax=1)
            ax.set_xlabel('Predicted label ($T_2$)', fontdict=font1)
            ax.set_ylabel('True label ($T_2$)', fontdict=font1)
        # cmap如: cividis, Purples, PuBu, viridis, magma, inferno; fmt: default=>'.2g'
    else:
        if task == 1:
            sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
            ax.set_xlabel('Predicted label ($T_1$)', fontdict=font1)
            ax.set_ylabel('True label ($T_1$)', fontdict=font1)
        elif task == 2:
            sns.heatmap(cm, annot=True, ax=ax, cmap='Greens', fmt='d')
            ax.set_xlabel('Predicted label ($T_2$)', fontdict=font1)
            ax.set_ylabel('True label ($T_2$)', fontdict=font1)
        # cmap如: plasma, viridis, magma, inferno, Pastel1_r; fmt: default=>'.2g'
    # ax.set_title('FTI-task', fontdict=font1)  # 标题

    # root = r'C:\Users\xiezongliang\Desktop\Multi-task Learning\插图'
    # f = os.path.join(root, fig_name)
    # plt.savefig(f, dpi=600)
    # print(f'Save at\n{f}')

    plt.show()

    # 注意在程序末尾加 plt.show()


def t_sne(input_data, input_label, classes, task, fig_name=None, labels=None, n_dim=2):
    input_label = input_label.astype(dtype=int)
    da = TSNE(n_components=n_dim, init='pca', random_state=0, angle=0.3).fit_transform(input_data)
    da = MinMaxScaler().fit_transform(da)  # (n, n_dim)

    figs = plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
    #  实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...

    if labels is None:
        if classes == 3 and task == 1:
            labels = ['NC', 'IF', 'OF']  # three types
        elif classes == 4 and task == 1:
            labels = ['NC', 'IF', 'OF', 'RF']  # four types
        elif classes == 4 and task == 2:
            labels = ['F0', 'F1', 'F2', 'F3']  # four types
    assert len(labels) == classes

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    ax = figs.add_subplot(111)
    # "husl", "muted"
    palette = np.array(sns.color_palette(palette="husl", n_colors=classes))  # [classes, 3]
    # print(palette.shape)
    sample_numbers = len(input_label)
    if classes == 3 and task == 1:
        type0 = np.empty((0, n_dim), dtype=np.float32)
        type1 = np.empty((0, n_dim), dtype=np.float32)
        type2 = np.empty((0, n_dim), dtype=np.float32)
        for i in range(sample_numbers):
            if input_label[i] == 0:
                type0 = np.vstack((type0, da[i]))
            elif input_label[i] == 1:
                type1 = np.vstack((type1, da[i]))
            elif input_label[i] == 2:
                type2 = np.vstack((type2, da[i]))
        ax.scatter(type0[:, 0], type0[:, 1], s=100, color=palette[0], alpha=0.8, marker=mark[0], label=labels[0])
        ax.scatter(type1[:, 0], type1[:, 1], s=100, color=palette[1], alpha=0.8, marker=mark[1], label=labels[1])
        ax.scatter(type2[:, 0], type2[:, 1], s=100, color=palette[2], alpha=0.8, marker=mark[2], label=labels[2])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', prop=font, labelspacing=1)
    elif classes == 4 and task == 1:
        type0 = np.empty((0, n_dim), dtype=np.float32)
        type1 = np.empty((0, n_dim), dtype=np.float32)
        type2 = np.empty((0, n_dim), dtype=np.float32)
        type3 = np.empty((0, n_dim), dtype=np.float32)
        for i in range(sample_numbers):
            if input_label[i] == 0:
                type0 = np.vstack((type0, da[i]))
            elif input_label[i] == 1:
                type1 = np.vstack((type1, da[i]))
            elif input_label[i] == 2:
                type2 = np.vstack((type2, da[i]))
            elif input_label[i] == 3:
                type3 = np.vstack((type3, da[i]))
        ax.scatter(type0[:, 0], type0[:, 1], s=100, color=palette[0], alpha=0.8, marker=mark[0], label=labels[0])
        ax.scatter(type1[:, 0], type1[:, 1], s=100, color=palette[1], alpha=0.8, marker=mark[1], label=labels[1])
        ax.scatter(type2[:, 0], type2[:, 1], s=100, color=palette[2], alpha=0.8, marker=mark[2], label=labels[2])
        ax.scatter(type3[:, 0], type3[:, 1], s=100, color=palette[3], alpha=0.8, marker=mark[3], label=labels[3])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', prop=font, labelspacing=1)
    elif classes == 4 and task == 2:
        type0 = np.empty((0, n_dim), dtype=np.float32)
        type1 = np.empty((0, n_dim), dtype=np.float32)
        type2 = np.empty((0, n_dim), dtype=np.float32)
        type3 = np.empty((0, n_dim), dtype=np.float32)
        for i in range(sample_numbers):
            if input_label[i] == 0:
                type0 = np.vstack((type0, da[i]))
            elif input_label[i] == 1:
                type1 = np.vstack((type1, da[i]))
            elif input_label[i] == 2:
                type2 = np.vstack((type2, da[i]))
            elif input_label[i] == 3:
                type3 = np.vstack((type3, da[i]))
        ax.scatter(type0[:, 0], type0[:, 1], s=100, color='#2a9d8f', alpha=0.8, marker=mark[0], label=labels[0])
        ax.scatter(type1[:, 0], type1[:, 1], s=100, color='#457b9d', alpha=0.8, marker=mark[1], label=labels[1])
        ax.scatter(type2[:, 0], type2[:, 1], s=100, color='#f4a261', alpha=0.8, marker=mark[2], label=labels[2])
        ax.scatter(type3[:, 0], type3[:, 1], s=100, color='#d62828', alpha=0.8, marker=mark[3], label=labels[3])
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-0.1, 1.3)
        ax.legend(loc='upper right', prop=font, labelspacing=1)

    # root = r'C:\Users\xiezongliang\Desktop\Multi-task Learning\插图'
    # path = os.path.join(root, fig_name)
    # plt.savefig(path, dpi=600)
    # print('Save t-SNE to \n', path)
    plt.show()

    # title = 't-SNE embedding of %s (time %.2fs)' % (name, (time() - t0))
    # plt.title(title)
    print('t-SNE Done!')
    return figs


def plot_fft(y, fs):
    """
    :param y: 原始一维振动信号
    :param fs: 采样频率
    :return:
    """
    t = np.array([i * 1 / fs for i in range(len(y))])
    # 原始信号
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('Time ($s$)')
    plt.ylabel('Amplitude ($Unit$)')
    plt.title('raw_data')

    N = len(y)  # 采样点
    T = 1.0 / fs  # 采样间距
    fft_y = fft(y)
    xf = fftfreq(N, T)[:N // 2]  # 取一半频率点

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模
    norm_y = (abs_y / N) * 2  # 归一化处理（双边频谱）
    yf = norm_y[:N // 2]  # 取一半
    plt.subplot(2, 1, 2)
    plt.plot(xf, yf)
    plt.xlabel('Frequency ($Hz$)')
    plt.ylabel('Amplitude ($Unit$)')
    plt.show()
