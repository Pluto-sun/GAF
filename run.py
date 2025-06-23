import argparse
import os
import torch
import torch.backends
from exp.exp import Exp
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='GAF Classification')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='classification',
                        help='task name, options:[classification]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='SAHU', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/SAHU/', help='root path of the data file')
    parser.add_argument('--step', type=int, required=True, help='slide step')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes for classification')
    parser.add_argument('--result_path', type=str, default='./result/', help='result path')
    parser.add_argument('--test_size', type=float, default=0.2, help='test size')
    # model define
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--gaf_method', type=str, default='summation', help='GAF method; summation or difference')
    parser.add_argument('--feature_dim', type=int, default=32, help='feature dimension')
    parser.add_argument('--data_type_method', type=str, default='float32', 
                        help='Data type conversion method; options: [float32, uint8, uint16]')
    # GNN
    parser.add_argument('--use_attention', type=bool, default=True, help='use attention')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--sample_size', type=int, default=10000, help='sample size')
    # 添加通道分组参数
    parser.add_argument('--channel_groups', type=str, default=None,
                        help='Channel grouping for ClusteredResNet. Format: "0,1,2|3,4,5|6,7,8"')
    parser.add_argument('--hvac_groups', type=str, default=None,
                        help='HVAC signal grouping for MultiImageFeatureNet. Format: "SA_TEMP,OA_TEMP|OA_CFM,RA_CFM|SA_SP"')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    
    # 处理通道分组参数
    if args.channel_groups is not None:
        try:
            # 将字符串格式的分组转换为列表格式
            args.channel_groups = [
                [int(x) for x in group.split(',')]
                for group in args.channel_groups.split('|')
            ]
        except Exception as e:
            print(f"Error parsing channel_groups: {e}")
            print("Using default channel grouping")
            args.channel_groups = None
    
    # 处理HVAC信号分组参数
    if args.hvac_groups is not None:
        try:
            # 将字符串格式的HVAC分组转换为列表格式
            args.hvac_groups = [
                [signal.strip() for signal in group.split(',')]
                for group in args.hvac_groups.split('|')
            ]
            print(f"HVAC Groups parsed: {len(args.hvac_groups)} groups")
            for i, group in enumerate(args.hvac_groups):
                print(f"  Group {i}: {group}")
        except Exception as e:
            print(f"Error parsing hvac_groups: {e}")
            print("Using default HVAC grouping")
            args.hvac_groups = None

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("torch.cuda.current_device():", torch.cuda.current_device())
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_sl{}_step{}_gaf{}_fd{}_dtype{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.step,
                args.gaf_method,
                args.feature_dim,
                args.data_type_method,
                args.des, ii)
            exp = Exp(args,setting)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train()
            exp.evaluate_report()

            print('>>>>>>>validating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.vali()
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_sl{}_step{}_gaf{}_fd{}_dtype{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.step,
            args.gaf_method,
            args.feature_dim,
            args.data_type_method,
            args.des, ii)

        print('>>>>>>>validating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
