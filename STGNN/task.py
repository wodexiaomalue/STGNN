import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import time
from model.model import Informer
import warnings
import os
import torch
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate
from dataset import Dataset_Custom, Dataset_Pred
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = self._build_model().to(self.device)

        # 是否加载参数
        # if args.load:
        #     best_model_path = args.path
        #     pretrained_dict = torch.load(best_model_path)
        #     model_dict = self.model.state_dict()
        #     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     # overwrite entries in the existing state dict
        #     model_dict.update(pretrained_dict)
        #     self.model.load_state_dict(model_dict)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _select_optimizer(self):
        raise NotImplementedError
        return None

    def _select_criterion(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Task(Basic):
    def __init__(self, args):
        super(Task, self).__init__(args)

    def _build_model(self):
        model = Informer(
            self.args.encoder_in,
            self.args.decoder_in,
            self.args.output,
            self.args.pred_len,
            self.args,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.distil,
            self.args.mix,
            self.args.output_attention,
        ).float()

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("numparam:", n_parameters)

        return model

    def _get_data(self, flag):
        args = self.args

        timeenc = 0 if args.embed != 'timeF' else 1

        Data = Dataset_Custom
        if flag == 'test':
            # drop_last 意思是数据的最后剩余的不够一个batch就丢掉
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 2
            # freq = args.detail_freq
            freq = args.freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            timeenc=timeenc,
            freq=freq,
            scale=True
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def train(self, ckp_name):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, ckp_name)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)  # 模型连续多少次不更新就停掉

        criterion = self._select_criterion()
        optimizer = self._select_optimizer()

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(train_loader):
                iter_count += 1

                optimizer.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_time, batch_y_time)  # 得到预测值和真实值
                loss = criterion(pred, true)  # 32 24 12
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(vali_loader):
            pred, true = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_time, batch_y_time)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_time, batch_y_time) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_time, batch_y_time)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_time, batch_y_time):

        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_time = batch_x_time.float().to(self.device)
        batch_y_time = batch_y_time.float().to(self.device)

        # decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()  # decoder输入以0为初始化
            # print(dec_inp.shape)
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()  # decoder输入以1为初始化
            # print(dec_inp.shape)

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # 32 ，72， 12

        # encoder - decoder
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_time, dec_inp, batch_y_time)[0]
        else:
            outputs = self.model(batch_x, batch_x_time, dec_inp, batch_y_time)
        # outputs, attn = self.model(batch_x, batch_x_time, dec_inp, batch_y_time)
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
            batch_y = dataset_object.inverse_transform(batch_y)

        return outputs, batch_y

    def predict(self, setting):
        pre_data, pre_loader = self._get_data(flag='pred')
        self.model.eval()
        # 读取 .npy 文件
        test_data = np.load('./data/ped_ground_truth_test.npy')
        test_data = pre_data.scaler.transform(test_data)

        test_data_tensor = torch.tensor(test_data)
        time_data = np.load('./data/ped_ground_truth.npy')
        time_data_tensor = torch.tensor(time_data)

        preds = []
        trues = []

        for idx in range(0, test_data_tensor.shape[0] - 1, 2):
            # data = test_data_tensor[idx,:,:].unsqueeze(0)
            # time = time_data_tensor[idx,:,:].unsqueeze(0)

            data1 = test_data_tensor[idx, :, :].unsqueeze(0)  # 第一个样本，维度变为 (1, T, F)
            data2 = test_data_tensor[idx + 1, :, :].unsqueeze(0)  # 第二个样本，维度变为 (1, T, F)

            # 拼接这两个样本，形成一个 batch，维度变为 (2, T, F)
            data = torch.cat((data1, data2), dim=0)

            time1 = time_data_tensor[idx, :, :].unsqueeze(0)  # 第一个时间样本
            time2 = time_data_tensor[idx + 1, :, :].unsqueeze(0)  # 第二个时间样本
            time = torch.cat((time1, time2), dim=0)

            batch_x = data[:,:14,:]
            batch_y = data
            batch_x_time = time[:,:14,:]
            batch_y_time = time
            pred, true = self._process_one_batch(
                pre_data, batch_x, batch_y, batch_x_time, batch_y_time)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        # preds = np.array(preds)
        # trues = np.array(trues)
        #
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # point71pre = preds[:,:,6]
        # point71tru = trues[:,:,6]
        # arr = point71pre[:211,0]
        # np.save('./results/inpre2410.npy', preds)
        # np.save('./results/intre2410.npy', trues)

        return
