import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from .metrics import metric
from tqdm import tqdm
from .tools import adjust_learning_rate, EarlyStopping
from torch.optim import lr_scheduler
import torch.nn as nn
import time


def model_train(args, model, model_optim, criterion, train_loader, epochs):
    train_steps = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    for ep in range(epochs+1):
        train_loss = []
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(args.device)

            batch_y = batch_y.float().to(args.device)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, res, trend = model(batch_x)
            else:
                outputs, res, trend = model(batch_x)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
            loss = criterion(outputs, batch_y)
            # a = res.cpu().detach().numpy()
            # b = trend.cpu().detach().numpy()
            # cos_sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
            # print('cos_sim:', cos_sim)
            train_loss.append(loss.item())

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, ep+1, args, printout=False)
                scheduler.step()

        train_loss = np.average(train_loss)
    return train_loss


def evaluate_synset(net, x_sys, y_sys, train, vail_loader, test_loader, args):
    lr = float(args.lr_ts)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr_ts))
    train_loader = TensorDataset(x_sys, y_sys)
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=16, shuffle=True, num_workers=0)
    criterion = nn.MSELoss().to(args.device)
    acc_train_list = []
    loss_train_list = []

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    start = time.time()
    path = os.path.join(args.checkpoints, args.save_path)
    if not os.path.exists(path):
        os.makedirs(path)
    for ep in range(Epoch + 1):
        loss_train = model_evaluate(args, net, optimizer, criterion, train_loader, ep)
        loss_train_list.append(loss_train)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        vail_loss, _, _, _ = model_test(args, net, optimizer, criterion, test_loader)
        early_stopping(vail_loss, net, path)
        if early_stopping.early_stop:
            break
    best_model_path = path + '/' + 'checkpoint.pth'
    net.load_state_dict(torch.load(best_model_path))

    time_train = time.time() - start

    with torch.no_grad():
        loss, mae, mse, rmse = model_test(args, net, optimizer, criterion, test_loader)
    print('%s Evaluate: epoch = %04d train time = %.4f s train loss = %.6f , test mae = %.4f, '
          'test mse = %.4f' % (
              get_time(), Epoch, time_train, loss_train, mae, mse))

    return net, acc_train_list, mae, mse


def model_test(args, model, model_optim, criterion, train_loader):
    preds = []
    trues = []
    total_loss = []
    model.eval()
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        model_optim.zero_grad()
        batch_x = batch_x.float().to(args.device)

        batch_y = batch_y.float().to(args.device)
        # print(torch.cuda.memory_allocated() / (1024 * 1024))
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, res, trend = model(batch_x)
        else:
            outputs, res, trend = model(batch_x)
        # print(torch.cuda.memory_allocated() / (1024 * 1024))
        f_dim = -1 if args.features == 'MS' else 0
        # print(outputs.shape,batch_y.shape)
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
        outputs = outputs.detach().cpu()
        batch_y = batch_y.detach().cpu()
        pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
        true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
        loss = criterion(pred, true)
        preds.append(pred.numpy())
        trues.append(true.numpy())
        total_loss.append(loss)
    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    total_loss = np.average(total_loss)
    mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
    return total_loss, mae, mse, rmse


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def model_evaluate(args, model, model_optim, criterion, train_loader, epoch):
    train_loss = []
    train_steps = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    model.train()
    for i, batch in enumerate(train_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        model_optim.zero_grad()
        batch_x = batch_x.float().to(args.device)

        batch_y = batch_y.float().to(args.device)
        # print("evaluate:",torch.cuda.memory_allocated() / (1024 * 1024))
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, res, trend = model(batch_x)
        else:
            outputs, res, trend = model(batch_x)
        # print("evaluate:",torch.cuda.memory_allocated() / (1024 * 1024))
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()

        adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
        scheduler.step()
    train_loss = np.average(train_loss)
    return train_loss


def streaming_learning(net, x_sys, y_sys, test_loader, args, train_loader_2, test_loader_2):
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    x = []
    y = []
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader_2)):
        x.append(batch_x)
        y.append(batch_y)
    x = torch.cat(x, 0)
    y = torch.cat(y, 0)
    x_sys = torch.cat((x_sys, x), dim=0)
    y_sys = torch.cat((y_sys, y), dim=0)
    train_loader = TensorDataset(x_sys, y_sys)
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=64, shuffle=True, num_workers=0)
    criterion = nn.MSELoss().to(args.device)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in range(Epoch + 1):
        loss_train = model_evaluate(args, net, optimizer, criterion, train_loader, ep)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                mae_1, mse_1, rmse_1 = model_test(args, net, optimizer, criterion, test_loader_2)
                mae_2, mse_2, rmse_2 = model_test(args, net, optimizer, criterion, test_loader)
                print('test on 30 precent dataset: Mae:%.6f,MSE:%.6f,RMSE:%.6f' % (mae_1, mse_1, rmse_1))
                print('test on 70 precent dataset: Mae:%.6f,MSE:%.6f,RMSE:%.6f' % (mae_2, mse_2, rmse_2))
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    return


def train_synset(args, model, model_optim, criterion, train_loader, test_loader, epochs):
    train_steps = len(train_loader)
    path = os.path.join(args.checkpoints, args.save_path)
    if not os.path.exists(path):
        os.makedirs(path)
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    epochs = epochs//200
    for ep in range(epochs+1):
        train_loss = []
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(args.device)

            batch_y = batch_y.float().to(args.device)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, res, trend = model(batch_x)
            else:
                outputs, res, trend = model(batch_x)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
            loss = criterion(outputs, batch_y)
            # a = res.cpu().detach().numpy()
            # b = trend.cpu().detach().numpy()
            # cos_sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
            # print('cos_sim:', cos_sim)
            train_loss.append(loss.item())

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, ep+1, args, printout=False)
                scheduler.step()

        train_loss = np.average(train_loss)
        vail_loss, _,_,_ = model_test(args, model, model_optim, criterion, test_loader)
        early_stopping(vail_loss, model, path)
        if early_stopping.early_stop:
            break
    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    return model, train_loss



class TensorDataset(Dataset):
    def __init__(self, batch_x, batch_y):  # images: n x c x h x w tensor
        self.batch_x = batch_x.detach().float()
        self.batch_y = batch_y.detach()

    def __getitem__(self, index):
        return self.batch_x[index], self.batch_y[index]

    def __len__(self):
        return self.batch_x.shape[0]
