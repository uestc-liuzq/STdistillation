import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
from process.data_factory import get_data
from process.exp import model_train, model_test
from network_patch import TSFE_Model
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)
    save_dir = os.path.join(args.buffer_path, args.data_name + '_' + str(args.pred_len))
    save_dir = os.path.join(save_dir, args.layers)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    ''' organize the real dataset '''
    train_data, train_loader = get_data(args, flag='train')
    test_data, test_loader = get_data(args, flag='test')
    criterion = nn.MSELoss().to(args.device)
    train_epochs = args.train_epochs
    trajectories = []

    for it in range(0, args.num_experts):

        ''' Train synthetic '''
        teacher_net = TSFE_Model(args)
        teacher_net = teacher_net.cuda()
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr)
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [train_epochs // 2 + 1]

        for e in range(train_epochs):
            train_loss = model_train(args, teacher_net, teacher_optim, criterion, train_loader,e)
            _, mae, mse, rmse = model_test(args, teacher_net, teacher_optim, criterion, test_loader)

            print("Itr: {}\tEpoch: {}\tTrain Loss: {}\nTest mae: {}\tTest mse: {}\tTest rmse: {}\t".format(it, e,
                                                                                                           train_loss,
                                                                                                           mae, mse,
                                                                                                           rmse))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            # if e in lr_schedule and args.decay:
            #     lr *= 0.1
            #     teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom,
            #                                     weight_decay=args.l2)
            #     teacher_optim.zero_grad()

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--num_experts', type=int, default=20, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.0001, help='learning rate for updating network parameters')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=2)

    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='layers id')
    parser.add_argument('--layers', type=str, default='TSFE',
                        help='layers name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--data_name', type=str, default='weather')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of layers checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=336, help='prediction sequence length')

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
    parser.add_argument('--enc_in', type=int, default=21,
                        help='encoder input size')  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of layers')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
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
    parser.add_argument('--devices', type=str, default='cuda:0', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See process/tools for usage')

    args = parser.parse_args()
    main(args)
