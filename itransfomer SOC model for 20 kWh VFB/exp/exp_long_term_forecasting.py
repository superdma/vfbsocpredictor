from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from torch import optim

import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
import os
#import psutil
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = optim.AdamW(self.model.parameters(),lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,seq_battery) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,seq_battery) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

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

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def fine_tune(self, setting,load_model_path):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print('loading model for fine tune')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + load_model_path, 'checkpoint.pth')))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,seq_battery) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

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

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model



    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            # 将setting字符串中test开头的部分截取掉
            setting_path = setting.split('_test')[0]
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting_path, 'checkpoint.pth')))

        preds = []
        trues = []
        battery = []
        #soc = []
        data_point = []
        cycle_index = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,seq_battery) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                seq_battery = seq_battery.float().to(self.device)
                #seq_soc = seq_soc.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                seq_battery = seq_battery[:, -self.args.pred_len:, :]
                #seq_soc = seq_soc[:, -self.args.pred_len:, :]


                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                seq_battery = seq_battery.detach().cpu().numpy()
                #seq_soc = seq_soc.detach().cpu().numpy()
                seq_data_point = batch_y[:, :, 0]    # data_point为Step_Time(s)列
                seq_cycle_index = batch_y[:, :, 1]  #cycle_index为Cycle_Index列

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                seq_battery = seq_battery[:, :, f_dim:]
                #seq_soc = seq_soc[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # 将seq_battery和seq_soc降成一维
                seq_battery = seq_battery.reshape(-1)
                #seq_soc = seq_soc.reshape(-1)
                seq_data_point = seq_data_point.reshape(-1)
                seq_cycle_index = seq_cycle_index.reshape(-1)

                preds.append(pred)
                trues.append(true)
                battery.append(seq_battery)
                #soc.append(seq_soc)
                data_point.append(seq_data_point)
                cycle_index.append(seq_cycle_index)

                # if i % 20 == 0:
                    # input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1]) 
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1]) 

        # 将preds和trues降成一维
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])

        mae, mse, mape, max_error = metric(preds, trues)
        print('Model mae:{:<22},mape:{:<22} mse:{:<22} max_error:{:<22}'.format(mae,mape,mse,max_error))

        battery_mae, battery_mse, battery_mape, battery_max_error = metric(battery, trues)
        print('Batry mae:{:<22},mape:{:<22} mse:{:<22} max_error:{:<22}'.format(battery_mae,battery_mape,battery_mse,battery_max_error))

        #soc_mae, soc_mse, soc_mape, soc_max_error = metric(soc, trues)
        #print('SoC   mae:{:<22},mape:{:<22} mse:{:<22} max_error:{:<22}'.format(soc_mae,soc_mape,soc_mse,soc_max_error))
        
        df_total_result = pd.DataFrame({'battery': [battery_mae,battery_mape,battery_mse,battery_max_error],
                                                     'Trans_model': [mae,mape,mse,max_error]}, index=['MAE', 'MAPE', 'MSE', 'Max Error']) 
                                                     # 'soc_model': [soc_mae,soc_mape,soc_mse,soc_max_error],
        df_total_result.columns.name = "全部数据"
        # df_total_result.to_csv(folder_path + 'metric_result.csv')

        # 将data_point,cycle_index,trues,battery,soc,preds拼接成一个dataframe
        data_point = np.array(data_point).reshape(-1)
        cycle_index = np.array(cycle_index).reshape(-1)
        trues = np.array(trues).reshape(-1)
        battery = np.array(battery).reshape(-1)
        #soc = np.array(soc).reshape(-1)
        preds = np.array(preds).reshape(-1)
        # 保存成csv文件
        df = pd.DataFrame({'data_point':data_point,'cycle_index':cycle_index,'true':trues,'battery':battery,'pred':preds})  #'soc':soc,
        df.to_csv(folder_path + 'result.csv')

        plt.figure(figsize=(16, 9),dpi=300)

        x = list(range(len(data_point)))
        plt.rcParams['font.size'] = 24
        plt.plot(x, trues,  'o', label='True Soc', color='#C4161B',  markersize=9)
        plt.plot(x, battery,  'v', label='OCV Soc', color='#FCAA8D', markersize=9)
        #plt.plot(x, soc, label='Model Predicted Soc', color='blue')
        plt.plot(x, preds,  '^', label='Transformer Soc', color='#1E6DB2',  markersize=9,alpha=0.8)
        plt.xlabel('Data point')
        plt.ylabel('SOC')
        plt.xlim(-100,len(x)+100)
        #plt.title('True SOC vs Predicted SOC')
        plt.legend(loc='best',frameon=False)
        # plt.show()
        plt.savefig(folder_path + 'result.jpg')

        df_3 = df
        df_8 = df

        # 取出df_3中true列小于等于0.3的行
        df_3 = df_3[df_3['true'] <= 0.3]
        # df_3.to_csv(folder_path + 'result_<0.3.csv')

        mae, mse, mape, max_error = metric(df_3['pred'], df_3['true'])
        battery_mae, battery_mse, battery_mape, battery_max_error = metric(df_3['battery'], df_3['true'])
        #soc_mae, soc_mse, soc_mape, soc_max_error = metric(df_3['soc'], df_3['true'])
        
        df_3_result = pd.DataFrame({'battery': [battery_mae,battery_mape,battery_mse,battery_max_error],                       
                           'Trans_model': [mae,mape,mse,max_error]}, index=['MAE', 'MAPE', 'MSE', 'Max Error'])
                           # 'soc_model': [soc_mae,soc_mape,soc_mse,soc_max_error],
        df_3_result.to_csv(folder_path + 'metric_result<0.3.csv')

        x = list(range(len(df_3)))
        plt.figure(figsize=(16, 9),dpi=300)
        plt.rcParams['font.size'] = 24
        plt.plot(x, df_3['true'],      'o', label='True Soc', color='#C4161B',  markersize=9)
        plt.plot(x, df_3['battery'], 'v', label='OCV Soc', color='#FCAA8D', markersize=9)
        #plt.plot(x, df_3['soc'], label='Model Predicted Soc', color='blue')
        plt.plot(x, df_3['pred'],  '^', label='Transformer Soc', color='#1E6DB2',  markersize=9,alpha=0.8)
        plt.xlabel('Data point')
        plt.ylabel('SOC')
        plt.xlim(-20,len(x)+20)
        #plt.title('True SOC vs Predicted SOC when true soc <= 0.3')
        plt.legend(loc='best',frameon=False)
        # plt.show()
        plt.savefig(folder_path + 'result0.3.jpg')

        # 取出df_8中true列大于等于0.8的行
        df_8 = df_8[df_8['true'] >= 0.8]
        # df_8.to_csv(folder_path + 'result_>0.8.csv')

        mae, mse, mape, max_error = metric(df_8['pred'], df_8['true'])
        battery_mae, battery_mse, battery_mape, battery_max_error = metric(df_8['battery'], df_8['true'])
        #soc_mae, soc_mse, soc_mape, soc_max_error = metric(df_8['soc'], df_8['true'])
        
        df_8_result = pd.DataFrame({'battery': [battery_mae,battery_mape,battery_mse,battery_max_error],                           
                           'Trans_model': [mae,mape,mse,max_error]}, index=['MAE', 'MAPE', 'MSE', 'Max Error'])
                           # 'soc_model': [soc_mae,soc_mape,soc_mse,soc_max_error],
        df_8_result.to_csv(folder_path + 'metric_result>0.8.csv')

        x = list(range(len(df_8)))
        plt.figure(figsize=(16, 9),dpi=300)
        plt.rcParams['font.size'] = 24
        plt.plot(x, df_8['true'],      'o', label='True Soc', color='#C4161B',  markersize=9)
        plt.plot(x, df_8['battery'],  'v', label='OCV Soc', color='#FCAA8D', markersize=9)
        #plt.plot(x, df_8['soc'], label='Model Predicted Soc', color='blue')
        plt.plot(x, df_8['pred'],  '^', label='Transformer Soc', color='#1E6DB2',  markersize=9,alpha=0.8)
        plt.xlabel('Data point')
        plt.ylabel('SOC')
        plt.xlim(-20,len(x)+20)
        plt.ylim((0.72,0.88))
        #plt.title('True SOC vs Predicted SOC when true soc >= 0.8')
        plt.legend(loc='best',frameon=False)
        # plt.show()
        plt.savefig(folder_path + 'result0.8.jpg')

        df_total_result = pd.concat([df_total_result, pd.DataFrame(index=['SOC < 0.3'])])
        df_total_result = pd.concat([df_total_result, df_3_result])
        df_total_result = pd.concat([df_total_result, pd.DataFrame(index=['SOC > 0.8'])])
        df_total_result = pd.concat([df_total_result, df_8_result])

        # 对csv所有元素居中
        df_total_result.style.set_properties(**{'text-align': 'center'})
        df_total_result.to_csv(folder_path + 'metric_result.csv')

        print('>>>>>>>Test Done!')

        return
