import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import recall_score,f1_score,precision_score
from .metrics import metric
from process.exp import TensorDataset
from .tools import adjust_learning_rate
from torch.optim import lr_scheduler
import torch.nn as nn
import time


def model_train(args, model, model_optim, criterion, train_loader, epoch):
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
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        model_optim.zero_grad()
        batch_x = batch_x.float().to(args.device)

        batch_y = batch_y.to(args.device)
        if args.use_amp:
            with torch.cuda.amp.autocast():
                outputs, res, trend = model(batch_x)
        else:
            outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()

        train_loss = np.average(train_loss)
        adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
        scheduler.step()
        return train_loss, model


def model_test(args, model, vali_loader):
    total_loss = []
    preds = []
    trues = []
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.to(args.device)

            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # decoder input
            # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in args.model or 'TST' in args.model:
                        outputs = model(batch_x)
            else:
                if 'Linear' in args.model or 'TST' in args.model:
                    outputs = model(batch_x)

            outputs = outputs.detach().cpu()
            batch_y = batch_y.detach().cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            outputs = outputs.numpy()
            batch_y = batch_y.numpy()
            outputs = np.argmax(outputs,axis=-1)
            preds.append(outputs)
            trues.append(batch_y)


            # loss = criterion(pred, true)
            # acciracy =

            # total_loss.append(loss)
    # total_loss = np.average(total_loss)
    accuracy = correct / total
    preds = np.array(preds)
    trues = np.array(trues)
    preds = [i[0] for i in preds]
    trues = [j[0] for j in trues]
    recall = recall_score(trues,preds,average='macro')
    precision = precision_score(trues,preds,average='macro')
    model.train()
    return accuracy,recall,precision


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def evaluate_synset(net, x_sys, y_sys, test_loader, args):
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    train_loader = TensorDataset(x_sys, y_sys)
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=32, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(args.device)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    for ep in range(Epoch + 1):
        loss_train,train_accuracy = model_evaluate(args, net, optimizer, criterion, train_loader, ep)
        loss_train_list.append(loss_train)
        if ep == Epoch:
            with torch.no_grad():
                test_accuracy,recall,precision = model_test(args, net, test_loader)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start

    print('%s Evaluate: epoch = %04d train time = %.4f s train loss = %.6f , test accuracy = %.4f test recall = %.4f test precision = %.4f' % (
        get_time(), Epoch, time_train, loss_train, test_accuracy,recall,precision))

    return net, acc_train_list,train_accuracy, test_accuracy


def model_evaluate(args, model, model_optim, criterion, train_loader, epoch):
    train_loss = []
    train_steps = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)
    total = 0
    correct = 0
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    model.train()
    for i, batch in enumerate(train_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        model_optim.zero_grad()
        batch_x = batch_x.float().to(args.device)

        batch_y = batch_y.to(args.device)
        # print("evaluate:",torch.cuda.memory_allocated() / (1024 * 1024))

        outputs= model(batch_x)
        # print("evaluate:",torch.cuda.memory_allocated() / (1024 * 1024))
        # f_dim = -1 if args.features == 'MS' else 0
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        outputs = outputs.detach().cpu()
        batch_y = batch_y.detach().cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)

        correct += (predicted == batch_y).sum().item()

        if args.use_amp:
            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            loss.backward()
            model_optim.step()

    train_loss = np.average(train_loss)
    adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
    scheduler.step()
    accuracy = correct / total
    model.train()
    return train_loss,accuracy
