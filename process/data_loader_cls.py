import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from process.timefeatures import time_features
import warnings
from tsai.all import *

warnings.filterwarnings('ignore')

class Dataset_cls(Dataset):
    def __init__(self, dsid, scale=True, mode='train'):
        # super().__init__()
        self.scale = StandardScaler()
        self.mode = mode
        self.dsid = dsid

        self.x, self.y, self.split = get_UCR_data(dsid=self.dsid, return_split=False)

        # print(self.x.shape, self.y.shape, self.split)
        self.len, self.channel, self.timesetp = self.x.shape

        # self.len = len(self.x)
        self.x = self.x.reshape(self.len, -1)
        # self.y = self.y.reshape(-1, 1)
        # print(self.y)
        if self.dsid == 'ECG200' or self.dsid == 'FordB':
            y_new = []
            for id, i in enumerate(self.y):
                i = int(i)
                if i == -1:
                    i = 1
                elif i == 1:
                    i = 2
                # print(i)
                y_new.append(i)
            self.y = np.array(y_new)
            # print(self.y.shape)

        self.scale.fit(self.x)
        self.x = self.scale.transform(self.x)

        self.x = self.x.reshape(self.len, self.timesetp, self.channel)

        # if self.mode == 'train':
        self.train_x = self.x[self.split[0]]
        self.train_y = self.y[self.split[0]]
        # print(self.train_x.shape)
        self.data_stamp_train = torch.ones(len(self.train_x), 4)
        # elif self.mode == 'val':
        self.val_x = self.x[self.split[1]]
        self.val_y = self.y[self.split[1]]
        self.data_stamp_val = torch.ones(len(self.val_x), 4)
        # elif self.mode == 'test':
        self.test_x = self.x[self.split[1]]
        self.test_y = self.y[self.split[1]]
        self.data_stamp_test = torch.ones(len(self.test_x), 4)

        # if self.dsid == 'ECG200':
        #     self.train_y = int(self.train_y) + 2
        #     self.val_y = int(self.val_y) + 2
        #     self.test_y = int(self.test_y) + 2
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_x)
        elif self.mode == 'val':
            return len(self.val_x)
        elif self.mode == 'test':
            return len(self.test_x)
    
    def __getitem__(self, index):
        if self.mode == 'train':
            # print(self.train_y[index].shape)
            # print(self.train_x[index].shape)
            # if self.dsid == 'ECG200':
            #     self.self.train_y[index] = int(self.train_y[index])+2
            return self.train_x[index], int(self.train_y[index])-1, self.data_stamp_train[index], self.data_stamp_train[index]
            # return self.train_x[index], self.train_y[index], None, None
        elif self.mode == 'val':
            # if self.dsid == 'ECG200':
            #     self.self.val_y[index] = int(self.val_y[index])+2
            return self.val_x[index], int(self.val_y[index])-1, self.data_stamp_val[index], self.data_stamp_val[index]
            # return self.val_x[index], self.val_y[index], None, None
        elif self.mode == 'test':
            # if self.dsid == 'ECG200':
            #     self.self.test_y[index] = int(self.test_y[index])+2
            return self.test_x[index], int(self.test_y[index])-1, self.data_stamp_test[index], self.data_stamp_test[index]
            # return self.test_x[index], self.test_y[index], None, None
        

def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    # print(args.dsid)

    data_set = Dataset_cls(dsid=args.dsid, mode=flag)
    # print(len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, num_workers=args.num_workers, drop_last=drop_last)
    # print(len(data_loader))

    return data_set, data_loader

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader