import os
import torch
import numpy as np
import visdom
import time
from torch.utils.data import DataLoader

from MTAGN_utils.train_utils import MyDataset, set_seed
from MTAGN_utils.plot_utils import plot_confusion_matrix
from data_generator.CWdata import CW_0, CW_1, CW_2, CW_3
from data_generator.SQdata import SQ_39, SQ_29, SQ_19
from data_generator.EBdata import EB
from models_STL.wdcnn import WDCNN
from models_STL.rescnn import ResCNN
from models.mtan_eca import STAN


# ==================== train and test =====================
class single_task_trainer:
    def __init__(self, model_):
        self.model = model_

    def train(self, train_loader_, valid_loader_, valid_set_, optimizer_, scheduler_, epochs=100, task=1):
        # ----------- model initial ----------
        loss_fuc = torch.nn.CrossEntropyLoss()

        print('--------------------Training--------------------')
        counter = 1
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            label = None

            self.model.train()
            for batch_idx, data in enumerate(train_loader_):
                optimizer_.zero_grad()
                inputs, label1, label2 = data
                inputs, label1, label2 = inputs.to(device), label1.to(device), label2.to(device)

                # forward + backward + update
                if task == 1:
                    label = label1
                elif task == 2:
                    label = label2
                output = self.model(inputs)
                loss = loss_fuc(output, label)
                train_loss += loss.item()

                loss.backward()
                optimizer_.step()

                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())
            train_acc = (train_acc / train_set.__len__()) * 100

            # valid
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0.0
                valid_acc = 0
                if epoch + 1 == epochs:
                    pred_label = np.array([], dtype=np.int64)
                for data in valid_loader_:
                    inputs, label1, label2 = data
                    inputs, label1, label2 = inputs.to(device), label1.to(device), label2.to(device)
                    output = self.model(inputs)

                    if task == 1:
                        label = label1
                    elif task == 2:
                        label = label2
                    loss = loss_fuc(output, label)
                    valid_loss += loss.item()

                    valid_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == label.cpu().numpy())

                    if epoch + 1 == epochs:
                        pred_label = np.append(pred_label, torch.max(output, dim=1)[1].cpu().numpy().astype('int64'))

                valid_acc = 100 * valid_acc / valid_set.__len__()

                # if epoch + 1 == EPOCHS and task == 1:
                #     plot_confusion_matrix(valid_set_.labels1.numpy(), pred_label, norm=False)
                #     plt.show()
                # elif epoch + 1 == EPOCHS and task == 2:
                #     plot_confusion_matrix(valid_set_.labels2.numpy(), pred_label, norm=False)
                #     plt.show()
            vis.line(Y=[[train_loss, valid_loss]], X=[counter], update=None if counter == 0 else 'append', win='loss',
                     opts=dict(legend=['train_loss', 'valid_loss'], title='Loss', ))
            vis.line(Y=[[train_acc, valid_acc]], X=[counter],
                     update=None if counter == 0 else 'append', win='accuracy',
                     opts=dict(legend=['train_acc', 'valid_acc'], title='Accuracy'))
            counter += 1
            print('epoch: [{}/{}] | Loss: {:.5f} | acc_tr: {:.2f}%'.format(epoch + 1, epochs, train_loss, train_acc))
        print('Finish training!')
        # order_save = input('Save model?(Y/N): ')
        # if order_save == 'Y' or order_save == 'y':
        self.save(filename=model_dir, model_name_pkl=model_name)

    def test(self, test_set_, test_loader_, task=1):
        self.load(model_path)
        self.model.eval()

        print('-------------------- Testing --------------------')
        acc = 0
        pred_label = np.array([], dtype=np.int64)
        with torch.no_grad():
            t0 = time.time()
            for data in test_loader_:
                inputs, label1, label2 = data
                inputs = inputs.to(device)

                if task == 1:
                    label = label1
                elif task == 2:
                    label = label2
                output = self.model(inputs)

                acc += np.sum(torch.max(output, dim=1)[1].cpu().numpy() == label.numpy())

                pred_label = np.append(pred_label, torch.max(output, dim=1)[1].cpu().numpy().astype('int64'))

            t1 = time.time()
            test_acc = 100 * acc / test_set_.__len__()
            test_time = t1 - t0

            # print('pred label:')
            # for i in range(7):
            #     print(pred_label[150 * i:150 * (i + 1)])
            print('Accuracy on test_dataset:')
            print(f'task{task}: {test_acc:.2f}%   [{acc}/{test_set_.__len__()}]')
            print(f'test time: {test_time:.4f}')
            # if task == 1:
            #     plot_confusion_matrix(test_set_.labels1.numpy(), pred_label, norm=False)
            #     plt.show()
            # elif task == 2:
            #     plot_confusion_matrix(test_set_.labels2.numpy(), pred_label, norm=False)
            #     plt.show()

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
    BATCH_SIZE = 64
    LR = 0.001
    # set_seed(2021)

    # define model, vis, optimiser
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    vis = visdom.Visdom(env='xzl_env')
    model = STAN(out_class=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    train_X, train_Y1, train_Y2, valid_X, valid_Y1, valid_Y2, test_X, test_Y1, test_Y2 = CW_3()
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

    trainer = single_task_trainer(model)

    # 模型名字及保存地址
    model_dir = r'F:\PycharmProject\MTNet\ablation experiments\multi-task learning'
    model_name = 'test.pkl'
    model_path = os.path.join(model_dir, model_name)
    print("Model path: ", model_dir)
    if not os.path.exists(model_dir):
        print(f'Root dir {model_dir} does not exit.')
        exit()
    else:
        print('File exist? ', os.path.exists(model_dir))
    order_tr = input('Train or not?(Y/N): ')
    if order_tr == 'Y' or order_tr == 'y':
        trainer.train(train_loader, valid_loader, valid_set, optimizer, scheduler, EPOCHS, task=2)

    order_te = input("Test or not?(Y/N): ")
    if order_te == "Y" or order_te == "y":
        trainer.test(test_set, test_loader, task=2)

