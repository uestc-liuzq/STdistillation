import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from PatchTST import Patch_Model
from process.data_factory import get_data
from process.exp import evaluate_synset, TensorDataset,streaming_learning
from utils import get_dataset, get_network, get_eval_pool, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def data_split(root_path,data_path):
    df = pd.read_csv(os.path.join(root_path,data_path))
    df_1 = df.iloc[:int(len(df)*0.7)]
    df_2 = df.iloc[int(len(df)*0.7):]
    df_1.to_csv(root_path+'train/'+data_path,index=False)
    df_2.to_csv(root_path+'test/'+data_path,index=False)
    return df_1, df_2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    # model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    model_eval_pool = ['PatchTST']

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation layers pool: ', model_eval_pool)

    ''' organize the real dataset '''
    root_path = args.root_path
    args.root_path = root_path+ 'train/'
    train_data, train_loader_1 = get_data(args, flag='train')
    test_data, test_loader_1 = get_data(args, flag='test')
    args.root_path = root_path+'test/'
    train_data_2,train_loader_2 = get_data(args, flag='train')
    test_data_2,test_loader_2 = get_data(args, flag='test')
    x = []
    y = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader_1)):
        x.append(batch_x)
        y.append(batch_y)
    x = torch.cat(x,dim=0)
    x_sys = x[:500]
    y = torch.cat(y,dim=0)
    y_sys = y[:500]
    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    ''' training '''
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_ts = torch.optim.SGD([x_sys], lr=args.lr_ts, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_ts.zero_grad()

    criterion = nn.MSELoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.data_name + '_' + str(args.pred_len))
    expert_dir = os.path.join(expert_dir, args.layers)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
            print(n)
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_mean_mae = {m: 0 for m in model_eval_pool}

    best_std_mae = {m: 0 for m in model_eval_pool}

    best_mean_mse = {m: 0 for m in model_eval_pool}

    best_std_mse = {m: 0 for m in model_eval_pool}

    for it in range(0, args.Iteration+1):

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.layers, model_eval, it))
                MAE = []
                MSE = []

                net_eval = Patch_Model(args).to(args.device) # get a random layers
                args.lr_net = syn_lr.item()
                batch_y = y_sys
                with torch.no_grad():
                    batch_x = x_sys
                x_syn_eval, y_syn_eval = copy.deepcopy(batch_x.detach()), copy.deepcopy(batch_y.detach())
                _,_,mae,mse,rmse = evaluate_synset(net_eval, x_syn_eval,y_syn_eval, test_loader_1, args)

                MAE.append(mae)
                MSE.append(mse)
                MAE = np.array(MAE)
                MSE = np.array(MSE)
                mae_test_mean = np.mean(MAE)
                mae_test_std = np.std(MAE)
                mse_test_mean = np.mean(MSE)
                mse_test_std = np.std(MSE)
                if mae_test_mean < best_mean_mae[model_eval] and mse_test_mean < best_mean_mse[model_eval]:
                    best_mean_mae[model_eval] = mae_test_mean
                    best_std_mae[model_eval] = mae_test_std
                    best_mean_mse[model_eval] = mse_test_mean
                    best_std_mse[model_eval] = mse_test_std
                # print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'MAE/{}'.format(model_eval): mae_test_mean}, step=it)
                wandb.log({'best MAE/{}'.format(model_eval): best_mean_mae[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): mae_test_std}, step=it)
                wandb.log({'best Std/{}'.format(model_eval): best_std_mae[model_eval]}, step=it)
                wandb.log({'MSE/{}'.format(model_eval): mse_test_mean}, step=it)
                wandb.log({'best MSE/{}'.format(model_eval): best_mean_mse[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): mse_test_std}, step=it)
                wandb.log({'best Std/{}'.format(model_eval): best_std_mse[model_eval]}, step=it)

                streaming_learning(net_eval, x_syn_eval, y_syn_eval, test_loader_1, args,train_loader_2,test_loader_2)



        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = Patch_Model(args).to(args.device)  # get a random layers

        student_net = ReparamModule(student_net)

        # if args.distributed:
        #     student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)


        param_loss_list = []
        param_dist_list = []
        indices_chunks = []
        train_loader = TensorDataset(x_sys[:100], y_sys[:100])
        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=2, shuffle=True, num_workers=0)
        for step, batch in enumerate(train_loader):
            x = batch[0].float().to(args.device)
            this_y = batch[1].float().to(args.device)

            # if args.distributed:
            #     forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            # else:
            forward_params = student_params[-1]
            x, res, trend= student_net(x, flat_param=forward_params)
            f_dim = -1 if args.features == 'MS' else 0
            x = x[:, -args.pred_len:, f_dim:]
            this_y = this_y[:, -args.pred_len:, f_dim:].to(args.device)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)


        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="mean")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="mean")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_ts.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_ts.step()
        optimizer_lr.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=50, help='epochs to train a layers with synthetic ECG200')
    parser.add_argument('--Iteration', type=int, default=1000, help='how many distillation steps to perform')

    parser.add_argument('--lr_ts', type=float, default=0.001, help='learning rate for updating synthetic data')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.0001, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.0001, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real ECG200')
    parser.add_argument('--batch_syn', type=int, default=int, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic ECG200')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--load_all', default=False, action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=2, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='layers id')
    parser.add_argument('--layers', type=str, default='PatchTST',
                        help='layers name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--data_name', type=str, default='traffic')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of layers checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # DLinear
    # parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='cuda', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See process/tools for usage')

    args = parser.parse_args()
    data_split(args.root_path,args.data_path)
    main(args)
