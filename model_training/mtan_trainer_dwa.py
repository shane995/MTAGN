import os
import torch
import numpy as np
import visdom
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from MTAGN_utils.train_utils import MyDataset, set_seed
from MTAGN_utils.plot_utils import plot_confusion_matrix
from data_generator.CWdata import CW_0, CW_1, CW_2, CW_3
from data_generator.SQdata import SQ_39, SQ_29, SQ_19
from data_generator.EBdata import EB
from models.jlcnn import JLCNN
from models.mt1dcnn import MT1DCNN
from models.mtagn import MTAGN


# ==================== train and test =====================
class multi_task_trainer:
    def __init__(self, model_):
        self.model = model_

    def train(self, train_loader_, valid_loader_, valid_set_, optimizer_, scheduler_, epochs=100):
        # ----------- model initial ----------
        loss_fuc = torch.nn.CrossEntropyLoss()

        train_batch = len(train_loader_)
        valid_batch = len(valid_loader_)

        print('--------------------Training--------------------')
        counter = 1
        T = 2
        avg_cost = np.zeros([epochs, 6])  # 0\1\2 train loss; 3\4\5 valid loss
        lambda_weight = np.ones([2, epochs])
        weight = np.zeros((100, 2))
        for epoch in range(epochs):
            train_loss_1 = 0.0
            train_loss_2 = 0.0
            train_loss = 0.0
            train_acc_1 = 0
            train_acc_2 = 0

            cost = np.zeros(6, dtype=np.float32)

            if epoch == 0 or epoch == 1:
                lambda_weight[:, epoch] = 1.0
            else:
                w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
                w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
                lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))

            self.model.train()
            for batch_idx, data in enumerate(train_loader_):
                optimizer_.zero_grad()
                inputs, label1, label2 = data
                inputs, label1, label2 = inputs.to(device), label1.to(device), label2.to(device)

                # forward + backward + update
                output1, output2 = self.model(inputs)
                tr_loss = [loss_fuc(output1, label1),
                           loss_fuc(output2, label2)]
                loss = sum(lambda_weight[i, epoch] * tr_loss[i] for i in range(2))

                train_loss_1 += tr_loss[0].item()
                train_loss_2 += tr_loss[1].item()
                train_loss += loss.item()

                loss.backward()
                optimizer_.step()

                cost[0] = tr_loss[0].item()
                cost[1] = tr_loss[1].item()
                cost[2] = loss.item()

                avg_cost[epoch, :3] += cost[:3] / train_batch

                train_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                train_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())
            train_acc_1 = (train_acc_1 / train_set.__len__()) * 100
            train_acc_2 = (train_acc_2 / train_set.__len__()) * 100

            # valid
            self.model.eval()
            with torch.no_grad():
                valid_loss_1 = 0.0
                valid_loss_2 = 0.0
                valid_loss = 0.0
                valid_acc_1 = 0
                valid_acc_2 = 0
                if epoch + 1 == epochs:
                    pred_label1 = np.array([], dtype=np.int64)
                    pred_label2 = np.array([], dtype=np.int64)
                for data in valid_loader_:
                    inputs, label1, label2 = data
                    inputs, label1, label2 = inputs.to(device), label1.to(device), label2.to(device)
                    output1, output2 = self.model(inputs)
                    val_loss = [loss_fuc(output1, label1),
                                loss_fuc(output2, label2)]
                    loss = sum(lambda_weight[i, epoch] * val_loss[i] for i in range(2))

                    valid_loss_1 += val_loss[0].item()
                    valid_loss_2 += val_loss[1].item()
                    valid_loss += loss.item()

                    cost[3] = val_loss[0].item()
                    cost[4] = val_loss[1].item()
                    cost[5] = loss.item()

                    avg_cost[epoch, 3:6] += cost[3:6] / valid_batch

                    valid_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                    valid_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())

                    if epoch + 1 == epochs:
                        pred_label1 = np.append(pred_label1, torch.max(output1, dim=1)[1].cpu().numpy().astype('int64'))
                        pred_label2 = np.append(pred_label2, torch.max(output2, dim=1)[1].cpu().numpy().astype('int64'))

                valid_acc_1 = 100 * valid_acc_1 / valid_set.__len__()
                valid_acc_2 = 100 * valid_acc_2 / valid_set.__len__()

                if epoch + 1 == epochs:
                    plot_confusion_matrix(valid_set_.labels1.numpy(), pred_label1, norm=False, task=1)
                    plot_confusion_matrix(valid_set_.labels2.numpy(), pred_label2, norm=False, task=2)
                    plt.show()
            # vis.line(Y=[[avg_cost[epoch, 0]*train_batch, avg_cost[epoch, 1]*train_batch, avg_cost[epoch, 2]*train_batch, avg_cost[epoch, 3]*train_batch,
            #              avg_cost[epoch, 4]*train_batch, avg_cost[epoch, 5]*train_batch]], X=[counter],
            #          update=None if counter == 0 else 'append', win='loss',
            #          opts=dict(legend=['train_loss_1', 'train_loss_2', 'train_loss', 'valid_loss_1', 'valid_loss_2',
            #                            'valid_loss'], title='Loss', ))
            vis.line(Y=[
                [avg_cost[epoch, 0] * train_batch, avg_cost[epoch, 1] * train_batch, avg_cost[epoch, 2] * train_batch]],
                X=[counter], update=None if counter == 0 else 'append', win='loss',
                opts=dict(legend=['train_loss_1', 'train_loss_2', 'train_loss', 'valid_loss_1', 'valid_loss_2',
                                  'valid_loss'], title='Loss', ))
            vis.line(Y=[[train_acc_1, train_acc_2, valid_acc_1, valid_acc_2]], X=[counter],
                     update=None if counter == 0 else 'append', win='accuracy',
                     opts=dict(legend=['train_acc_1', 'train_acc_2', 'valid_acc_1', 'valid_acc_2'], title='Accuracy'))
            vis.line(Y=[[lambda_weight[0, epoch], lambda_weight[1, epoch]]], X=[counter],
                     update=None if counter == 0 else 'append', win='weight',
                     opts=dict(legend=['weight1', 'weight2'], title='Weight'))
            counter += 1

            scheduler_.step()
            print(
                'epoch: [{}/{}] | Loss: {:.5f} | FTI_acc_tr: {:.2f}% | FSI_acc_tr: {:.2f}% | w1:{:.3f} w2:{:.3f}'.format(
                    epoch + 1, epochs, train_loss, train_acc_1, train_acc_2, lambda_weight[0, epoch],
                    lambda_weight[1, epoch]))
            weight[epoch, 0], weight[epoch, 1] = lambda_weight[0, epoch], lambda_weight[1, epoch]
        print('Finish training!')
        loss_out = avg_cost[:, :3] * train_batch
        print(loss_out.shape)
        np.savetxt('EB-train_loss-DWA.csv', loss_out, delimiter=',')
        print(weight.shape)
        print(weight)
        np.savetxt('lambda-weight-DWA.csv', weight, delimiter=',')
        order_save = input('Save model?(Y/N): ')
        if order_save == 'Y' or order_save == 'y':
            self.save(filename=model_dir, model_name_pkl=model_name)

    def test(self, test_set_, test_loader_):
        self.load(model_path)
        self.model.eval()

        print('-------------------- Testing --------------------')
        acc_1 = 0
        acc_2 = 0
        pred_label1 = np.array([], dtype=np.int64)
        pred_label2 = np.array([], dtype=np.int64)
        with torch.no_grad():
            t0 = time.time()
            for data in test_loader_:
                inputs, label1, label2 = data
                inputs = inputs.to(device)
                output1, output2 = self.model(inputs)

                acc_1 += np.sum(torch.max(output1, dim=1)[1].cpu().numpy() == label1.numpy())
                acc_2 += np.sum(torch.max(output2, dim=1)[1].cpu().numpy() == label2.numpy())

                pred_label1 = np.append(pred_label1, torch.max(output1, dim=1)[1].cpu().numpy().astype('int64'))
                pred_label2 = np.append(pred_label2, torch.max(output2, dim=1)[1].cpu().numpy().astype('int64'))
                t1 = time.time()

            FTI_acc = 100 * acc_1 / test_set_.__len__()
            FSI_acc = 100 * acc_2 / test_set_.__len__()
            te_time = t1 - t0

            # print('pred label1:')
            # for i in range(10):
            #     print(pred_label1[150 * i:150 * (i + 1)])
            # print('pred label2:')
            # for i in range(7):
            #     print(pred_label2[150 * i:150 * (i + 1)])
            print('Accuracy on test_dataset:')
            print(f'FTI-task: {FTI_acc:.2f}%   [{acc_1}/{test_set_.__len__()}]')
            print(f'FSI-task: {FSI_acc:.2f}%   [{acc_2}/{test_set_.__len__()}]')
            print(f'Test time: {te_time:.4f}')
            plot_confusion_matrix(test_set_.labels1.numpy(), pred_label1, norm=False, task=1)
            plot_confusion_matrix(test_set_.labels2.numpy(), pred_label2, norm=False, task=2)
            plt.show()

    def save(self, filename, model_name_pkl):
        if os.path.exists(filename):
            filename = os.path.join(filename, model_name_pkl)
        torch.save(self.model.state_dict(), filename)
        print(f'This model is saved at: {filename}')

    def load(self, filename):
        state = torch.load(filename)
        self.model.load_state_dict(state)
        print('Load model successfully from [%s]' % filename)


if __name__ == '__main__':
    # ==================== Hyper parameters =====================
    EPOCHS = 100
    BATCH_SIZE = 50
    LR = 0.001
    # set_seed(2021)

    # define model, vis, optimiser
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom(env='dwa')
    model = MTAGN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2 = CW_0()
    # ----------- train data ----------
    train_set = MyDataset(train_X, train_Y1, train_Y2)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    print('train_set')
    print("train_X:", train_X.shape)
    print("train_Y1:", train_Y1.shape)
    print("train_Y2:", train_Y2.shape)
    # ----------- valid data ----------
    valid_set = MyDataset(valid_X, valid_Y1, valid_Y2)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=BATCH_SIZE)
    print('valid_set')
    print("valid_X:", valid_X.shape)
    print("valid_Y1:", valid_Y1.shape)
    print("valid_Y2:", valid_Y2.shape)
    # ----------- data ----------
    test_set = MyDataset(test_X, test_Y1, test_Y2)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE)
    print("test_X:", test_X.shape)
    print("test_Y1:", test_Y1.shape)
    print("test_Y2:", test_Y2.shape)

    trainer = multi_task_trainer(model)

    # 模型名字及保存地址
    model_dir = r'F:\PycharmProject\MTNet\ablation experiments'
    model_name = 'mtan_eca6_dwa-cw0-100-3.pkl'
    # model_name = 'test.pkl'
    model_path = os.path.join(model_dir, model_name)
    print("Model path: ", model_dir)
    if not os.path.exists(model_dir):
        print(f'Root dir {model_dir} does not exit.')
        exit()
    else:
        print('File exist? ', os.path.exists(model_dir))
    # order_tr = input('Train or not?(Y/N): ')
    # if order_tr == 'Y' or order_tr == 'y':
    trainer.train(train_loader, valid_loader, valid_set, optimizer, scheduler, EPOCHS)

    order_te = input("Test or not?(Y/N): ")
    if order_te == "Y" or order_te == "y":
        trainer.test(test_set, test_loader)
