import torch
from torch.utils.data import TensorDataset
import scipy.io as scio
from sklearn import preprocessing
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset

def get_UCI2(path='datasets\\handwritten.mat', batchsize=200):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = min_max_scaler.fit_transform(data['X'][0][0])
    x2 = min_max_scaler.fit_transform(data['X'][0][1])
    x3 = min_max_scaler.fit_transform(data['X'][0][2])
    x4 = min_max_scaler.fit_transform(data['X'][0][3])
    x5 = min_max_scaler.fit_transform(data['X'][0][4])
    x6 = min_max_scaler.fit_transform(data['X'][0][5])
    y = data['Y']

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(y).cuda())

    return dataset

def get_UCI6(path = 'datasets\\handwritten.mat',batchsize=200):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = min_max_scaler.fit_transform(data['X'][0][0])
    x2 = min_max_scaler.fit_transform(data['X'][0][1])
    x3 = min_max_scaler.fit_transform(data['X'][0][2])
    x4 = min_max_scaler.fit_transform(data['X'][0][3])
    x5 = min_max_scaler.fit_transform(data['X'][0][4])
    x6 = min_max_scaler.fit_transform(data['X'][0][5])
    y = data['Y']

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x3 = x3.astype(np.float32)
    x4 = x4.astype(np.float32)
    x5 = x5.astype(np.float32)
    x6 = x6.astype(np.float32)

    train = TensorDataset(torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(),
                          torch.from_numpy(x4).cuda(), torch.from_numpy(x5).cuda(), torch.from_numpy(x6).cuda(), torch.from_numpy(y).cuda())

    return train

def get_Caltech101(path = 'datasets/Caltech101.mat',batchsize=200):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = min_max_scaler.fit_transform(data['fea'][0][0])
    x2 = min_max_scaler.fit_transform(data['fea'][0][1])
    x3 = min_max_scaler.fit_transform(data['fea'][0][2])
    x4 = min_max_scaler.fit_transform(data['fea'][0][3])
    x5 = min_max_scaler.fit_transform(data['fea'][0][4])
    x6 = min_max_scaler.fit_transform(data['fea'][0][5])
    y = data['gt']

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x3 = x3.astype(np.float32)
    x4 = x4.astype(np.float32)
    x5 = x5.astype(np.float32)
    x6 = x6.astype(np.float32)

    train = TensorDataset(torch.from_numpy(x5).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(),
                          torch.from_numpy(x4).cuda(), torch.from_numpy(x1).cuda(), torch.from_numpy(x6).cuda(), (torch.from_numpy(y)-1).cuda())

    return train

def get_MSRC(path = 'datasets\\MSRC_v1.mat',batchsize=200):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = min_max_scaler.fit_transform(data['fea'][0][0])
    x2 = min_max_scaler.fit_transform(data['fea'][0][1])
    x3 = min_max_scaler.fit_transform(data['fea'][0][2])
    x4 = min_max_scaler.fit_transform(data['fea'][0][3])
    x5 = min_max_scaler.fit_transform(data['fea'][0][4])
    y = data['gt']

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x3 = x3.astype(np.float32)
    x4 = x4.astype(np.float32)
    x5 = x5.astype(np.float32)
    # y = y.astype(np.float32)

    train = TensorDataset(torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(),torch.from_numpy(x4).cuda(),
                          torch.from_numpy(x5).cuda(), (torch.from_numpy(y)-1).cuda())

    return train

def get_Yale(path = 'datasets/Yale.mat', batchsize=200):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler()
    x1 = min_max_scaler.fit_transform(data['fea'][0][0])
    x2 = min_max_scaler.fit_transform(data['fea'][0][1])
    x3 = min_max_scaler.fit_transform(data['fea'][0][2])
    y = data['gt'] - 1

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x3 = x3.astype(np.float32)
    # y = y.astype(np.float32)

    train = TensorDataset(torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(), torch.from_numpy(y).cuda())

    return train


def get_NUS5(path='datasets/NUSWIDEOBJ.mat'):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x1 = min_max_scaler.fit_transform(data['X'][0][0])
    x2 = min_max_scaler.fit_transform(data['X'][0][1])
    x3 = min_max_scaler.fit_transform(data['X'][0][2])
    x4 = min_max_scaler.fit_transform(data['X'][0][3])
    x5 = min_max_scaler.fit_transform(data['X'][0][4])
    y = data['Y']

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x3 = x3.astype(np.float32)
    x4 = x4.astype(np.float32)
    x5 = x5.astype(np.float32)

    train = TensorDataset(torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(),
                          torch.from_numpy(x4).cuda(), torch.from_numpy(x5).cuda(), (torch.from_numpy(y)-1).cuda())

    return train

def get_ALOI(path='datasets\\ALOI_1K.mat'):
    data = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x1 = min_max_scaler.fit_transform(data['fea'][0][0])
    x2 = min_max_scaler.fit_transform(data['fea'][0][1])
    x3 = min_max_scaler.fit_transform(data['fea'][0][2])
    x4 = min_max_scaler.fit_transform(data['fea'][0][3])
    y = data['Y']

    x1 = x1.astype(np.float32)
    x2 = x2.astype(np.float32)
    x3 = x3.astype(np.float32)
    x4 = x4.astype(np.float32)

    train = TensorDataset(torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(),
                          torch.from_numpy(x4).cuda(), (torch.from_numpy(y)-1).cuda())

    return train

class NUS5(Dataset):
    def __init__(self, path='datasets\\NUSWIDEOBJ.mat'):
        super(NUS5, self).__init__()
        self.data = scio.loadmat(path)
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.x1 = self.min_max_scaler.fit_transform(self.data['X'][0][0])
        self.x2 = self.min_max_scaler.fit_transform(self.data['X'][0][1])
        self.x3 = self.min_max_scaler.fit_transform(self.data['X'][0][2])
        self.x4 = self.min_max_scaler.fit_transform(self.data['X'][0][3])
        self.x5 = self.min_max_scaler.fit_transform(self.data['X'][0][4])
        self.y = self.data['Y']
        self.classes_for_all_imgs = self.y-1

    def __getitem__(self, item):
        x1 = self.x1[item].astype(np.float32)
        x2 = self.x2[item].astype(np.float32)
        x3 = self.x3[item].astype(np.float32)
        x4 = self.x4[item].astype(np.float32)
        x5 = self.x5[item].astype(np.float32)
        return [torch.from_numpy(x1).cuda(), torch.from_numpy(x2).cuda(), torch.from_numpy(x3).cuda(),\
               torch.from_numpy(x4).cuda(), torch.from_numpy(x5).cuda(), (torch.from_numpy(self.y[item])-1).cuda()]

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs

if __name__ == '__main__':
    dataset = get_UCI6()

    dataLoader = DataLoader(dataset, batch_size=20000, shuffle=False, drop_last=False, pin_memory=False)
    length = 7

    for data in dataLoader:
        for i in range(length):
            print(torch.max(data[i]), torch.min(data[i]))

