import numpy as np

import torch
from torch.utils.data import Dataset

from einops import rearrange, repeat


class vit_neural_dataset(Dataset):
    """return data [b n] with a single label (100ms and its label)"""
    def __init__(self, dataset, label):
        self.data = torch.FloatTensor(dataset)
        self.label = torch.LongTensor(label)

        _, t, _ = self.data.shape
        self.data = rearrange(self.data, 'b t n -> (b t) n')
        self.label = self.label[:, None]
        self.label = repeat(self.label, 'b () -> b t', t=t)
        self.label = rearrange(self.label, 'b t -> (b t)')
        self.t_amount = t

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

    @property
    def cov(self):
        return np.ma.corrcoef(np.ma.masked_invalid(self.data.T))

    @property
    def ori_data(self):
        return self.data

class vit_neural_temp_dataset(Dataset):
    def __init__(self, dataset, label, scale_data=True, limited=False, limited_6=False, neuron_set=None):
        self.data = torch.FloatTensor(dataset)
        self.data_with_temp = self.data.clone()  # b t n

        if scale_data:
            self.data_with_temp = self._scale_data(self.data_with_temp)
            #...

        self.label = torch.LongTensor(label)

        _, t, _ = self.data.shape
        self.label = self.label[:, None]
        self.label = repeat(self.label, 'b () -> b t', t=t)
        self.label_with_temp = self.label.clone() # b t

        if limited:
            """default dynamic is 2"""
            self.data_new = []
            self.label_new = []
            for start in [0, 1, 2, 3, 4, 5]:
                self.data_new.append(self.data_with_temp[:, start:int(start+2), :])
                self.label_new.append(self.label_with_temp[:, start:int(start+2)])

            self.data_with_temp = torch.cat(self.data_new)
            self.label_with_temp = torch.cat(self.label_new)
        if limited_6:
            self.data_new = []
            self.label_new = []
            for start in [0, 1, 2]:
                self.data_new.append(self.data_with_temp[:, start:int(start+6), :])
                self.label_new.append(self.label_with_temp[:, start:int(start+6)])

            self.data_with_temp = torch.cat(self.data_new)
            self.label_with_temp = torch.cat(self.label_new)

        if (neuron_set[1] is not None) and (neuron_set is not None):
            if neuron_set[0] is True:
                self.data_with_temp = torch.cat(self.data_new)[:, :, :neuron_set[1]]
            else:
                self.data_with_temp = torch.cat(self.data_new)[:, :, -neuron_set[1]:]


    def __getitem__(self, index):
        """data_with_temp shape like [b t n], label_with_temp shape like [b t]"""
        return self.data_with_temp[index], self.label_with_temp[index]
        #return self.data_with_temp[index], self.label_with_temp[index, 0]

    def __len__(self):
        return self.data_with_temp.shape[0]

    @staticmethod
    def _scale_data(data):
        # data of shape (b t) n
        b, t, _ = data.shape
        data = rearrange(data, 'b t n -> (b t) n')
        bt, n = data.shape

        mean = torch.mean(data, axis=0).repeat(bt, 1)
        data = (data - mean)

        data = rearrange(data, '(b t) n -> b t n', b=b)
        return data

class direction_dataset(Dataset):
    def __init__(self, train_set, test_set, select_dir, comb=True, test_limited=False):
        self.select_dir = select_dir

        """data_with_temp shape like [b t n], label_with_temp shape like [b t]"""
        if comb:
            self.data = torch.cat([train_set.data_with_temp, test_set.data_with_temp], dim=0)
            self.label = torch.cat([train_set.label_with_temp, test_set.label_with_temp], dim=0)
        else:
            assert test_set == None
            self.data = train_set.data_with_temp
            self.label = train_set.label_with_temp

        dir_mask = self.label[:, 0]
        dir_mask_posi = [i for i in range(dir_mask.shape[0]) if (dir_mask[i] in select_dir)]

        #print(self.data.shape)
        #print(dir_mask_posi)

        self.data = self.data[dir_mask_posi]
        self.label = self.label[dir_mask_posi]

        if select_dir[0] != 0:
            new_label = torch.zeros(self.label.shape)
            for i in range(1, len(select_dir)):
                new_label[self.label == select_dir[i]] = i
            self.label = new_label.long()

        if test_limited:
            assert NotImplementedError('change this')
            print(self.data.shape)
            self.data[1, :] = self.data[0, :]
            # if x.shape[1] == 2:
            # x[:, 1, :] = x[:, 0, :]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

class time_dataset(Dataset):
    def __init__(self, train_set, test_set, train=True, comb=True, test_limited=False):
        """data_with_temp shape like [b t n], label_with_temp shape like [b t]"""
        if comb:
            self.data = torch.cat([train_set.data_with_temp, test_set.data_with_temp], dim=0)
            self.label = torch.cat([train_set.label_with_temp, test_set.label_with_temp], dim=0)
        else:
            assert test_set == None
            self.data = train_set.data_with_temp
            self.label = train_set.label_with_temp

        if train:
            self.data = self.data[:, :4, :]
            self.label = self.label[:, :4]
        else:
            self.data = self.data[:, 4:, :]
            self.label = self.label[:, 4:]

        # print(self.data.shape)

        # set them limited here
        limited = True
        if limited:
            """default dynamic is 2"""
            self.data_new = []
            self.label_new = []
            for start in [0, 1, 2]:
                self.data_new.append(self.data[:, start:int(start+2), :])
                self.label_new.append(self.label[:, start:int(start+2)])

            self.data_with_temp = torch.cat(self.data_new)
            self.label_with_temp = torch.cat(self.label_new)

        if test_limited:
            pass
            # self.data_with_temp[1, :] = self.label_with_temp[0, :]
            # if x.shape[1] == 2:
            # x[:, 1, :] = x[:, 0, :]

    def __getitem__(self, index):
        return self.data_with_temp[index], self.label_with_temp[index]

    def __len__(self):
        return self.data_with_temp.shape[0]

class multiple_datasets_list(Dataset):
    "datasets is a list of datasets"
    def __init__(self, datasets):
        self.datasets = datasets

        self.amount = len(datasets)
        self.lens = [data_i.__len__() for data_i in datasets]
        self.added_lens = [0]
        for i in range(self.amount):
            self.added_lens.append(self.added_lens[i] + self.lens[i])

        # print(self.lens)
        # print(self.added_lens)
        # [127, 144, 167, 172]
        # [0, 127, 271, 438, 610]
        # [32, 36, 42, 43]
        # [0, 32, 68, 110, 153]

    def __getitem__(self, index):
        for i, added_i in enumerate(self.added_lens):
            if index < added_i:
                return self.datasets[i-1].__getitem__(index-self.added_lens[i-1])

    def __len__(self):
        return sum(self.lens)

class multiple_datasets_cat(Dataset):
    """returns batches between four different animals together"""
    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(d[index] for d in self.datasets)
        # return self.datasets[i-1].__getitem__(index-self.added_lens[i-1])

    def __len__(self):
        return min(len(d) for d in self.datasets)