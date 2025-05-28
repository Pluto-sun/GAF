from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from models import RestNet, ClusteredResNet, VGGNet, ClusteredVGGNet,ClusteredInception
from exp.prune_model import  loss_with_regularization
from models.ClusteredResNet import global_prune
warnings.filterwarnings('ignore')

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        # 保存参数
        self.args = args
        self.model_dict = {
            'RestNet': RestNet,
            'ClusteredResNet': ClusteredResNet,
            'VGG': VGGNet,
            'ClusteredVGGNet': ClusteredVGGNet,
            'ClusteredInception': ClusteredInception
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba
        
        # 设置设备
        self.device = self._acquire_device()
        
        # 初始化数据加载器
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        self.test_data, self.test_loader = self._get_data(flag='test')
        
        # 构建并移动模型到设备
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # 使用已创建的训练数据加载器
        sample_data, _ = next(iter(self.train_loader))
        print(f"Sample data shape: {sample_data.shape}")
        self.args.seq_len = sample_data.shape[1]  # 时间步长
        self.args.enc_in = sample_data.shape[3]   # 特征维度
        
        # 使用args中的num_class参数
        self.args.num_class = self.args.num_class
        
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)
                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # ====== Step 1: 正则化训练（L1作用于BN gamma） ======
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label) in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                
                if i % 100 == 0:
                    # print(f"\nBatch {i+1}/{len(self.train_loader)}")
                    # print(f"Batch shape: {batch_x.shape}")
                    # print_gpu_memory()
                    
                    # 每100个批次清理一次GPU内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                outputs = self.model(batch_x)
                # 用L1正则化损失
                # loss = loss_with_regularization(self.model, outputs, label.long().squeeze(-1), lambda_l1=0.001)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(self.vali_data, self.vali_loader, criterion)
            test_loss, test_accuracy = self.vali(self.test_data, self.test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            
            # 每个epoch结束后清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # print("Epoch结束后清理GPU内存")
                # print_gpu_memory()
            
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # ====== Step 2: 剪枝 group_convs ======
        # print("\n=== Pruning group_convs after main training ===")
        # self.model = prune_group_convs(self.model, prune_ratio=0.2, device=self.device)
        # self.model = prune_merge_net(self.model, prune_ratio=0.2, device=self.device)
        # 全局剪枝（剪除gamma最小的20%通道）
        # self.model = global_prune(self.model, prune_ratio=0.2)
        # print("group_convs pruned.")

        # # ====== Step 3: 微调finetune（常规损失） ======
        # print("\n=== Finetuning pruned model ===")
        # finetune_epochs = 20
        # finetune_lr = self.args.learning_rate * 0.1
        # model_optim = torch.optim.RAdam(self.model.parameters(), lr=finetune_lr)
        # criterion = nn.CrossEntropyLoss()

        # # 新增：finetune early stopping
        # path_finetune = path + '_finetune'
        # if not os.path.exists(path_finetune):
        #     os.makedirs(path_finetune)
        # early_stopping_finetune = EarlyStopping(patience=4, verbose=True)

        # for epoch in range(finetune_epochs):
        #     self.model.train()
        #     train_loss = []
        #     for i, (batch_x, label) in enumerate(self.train_loader):
        #         model_optim.zero_grad()
        #         batch_x = batch_x.float().to(self.device)
        #         label = label.to(self.device)
        #         outputs = self.model(batch_x)
        #         loss = criterion(outputs, label.long().squeeze(-1))
        #         train_loss.append(loss.item())
        #         loss.backward()
        #         nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
        #         model_optim.step()
        #     train_loss = np.average(train_loss)
        #     vali_loss, val_accuracy = self.vali(self.vali_data, self.vali_loader, criterion)
        #     print(f"[Finetune] Epoch {epoch+1}/{finetune_epochs} | Train Loss: {train_loss:.3f} | Vali Loss: {vali_loss:.3f} | Vali Acc: {val_accuracy:.3f}")
        #     if torch.cuda.is_available():
        #         torch.cuda.empty_cache()
        #     # early stopping
        #     early_stopping_finetune(-val_accuracy, self.model, path_finetune)
        #     if early_stopping_finetune.early_stop:
        #         print("Finetune Early stopping")
        #         break

        # # 加载finetune最优模型
        # best_finetune_model_path = path_finetune + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_finetune_model_path))
        # print("Finetune finished and best model loaded.")
        return self.model

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(self.vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
