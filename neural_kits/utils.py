import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from neural_kits.neural_dataset import ReachNeuralDataset
from neural_kits.neural_datasets_additional import vit_neural_dataset, \
    vit_neural_temp_dataset, multiple_datasets_list, multiple_datasets_cat, direction_dataset, time_dataset

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers.
    Args:
        random_seed: Desired random seed.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

class get_animal_data(object):
    def __init__(self, time_select=8, binning_size=0.1, batch_size=128):
        self.time_select = time_select
        self.binning_size = binning_size
        self.batch_size = batch_size
        self.time_cut = int(time_select * (0.1 / binning_size))

    def single_animal_dataset(self, animal, day, type='temp', neuron_set=None, neuron_set_train=False):

        dataset = ReachNeuralDataset('./datasets/mihi-chewie',
                                     primate=animal,
                                     day=day,
                                     binning_period=self.binning_size,
                                     binning_overlap=0.0,
                                     scale_firing_rates=False,
                                     train_split=0.8)

        dataset.train()
        original_data_train, original_label_train = dataset.original_data, dataset.original_label
        train_data, train_label = dataset._same_length_for_psth(original_data_train, original_label_train)

        time_dim_append = np.zeros(shape=(train_data.shape[0], 8, train_data.shape[2]))
        train_data = np.concatenate([train_data, time_dim_append], axis=1)

        train_data = train_data[:, :self.time_cut, :]
        train_label = train_label[:, 0, 0]

        dataset.test()
        original_data_test, original_label_test = dataset.original_data, dataset.original_label
        test_data, test_label = dataset._same_length_for_psth(original_data_test, original_label_test)

        time_dim_append = np.zeros(shape=(test_data.shape[0], 8, test_data.shape[2]))
        test_data = np.concatenate([test_data, time_dim_append], axis=1)

        test_data = test_data[:, :self.time_cut, :]
        test_label = test_label[:, 0, 0]
        # print(test_data.shape) # (42, 17, 163)

        random_order = torch.randperm(test_data.shape[-1])
        test_data = test_data[:, :, random_order]
        train_data = train_data[:, :, random_order]

        if type == 'temp':
            train_set = vit_neural_temp_dataset(dataset=train_data, label=train_label)
            test_set = vit_neural_temp_dataset(dataset=test_data, label=test_label)
        elif type == 'limited':
            train_set = vit_neural_temp_dataset(dataset=train_data, label=train_label, limited=True, neuron_set=(neuron_set_train, neuron_set))
            test_set = vit_neural_temp_dataset(dataset=test_data, label=test_label, limited=True, neuron_set=(neuron_set_train, neuron_set))
        elif type == 'limited_6':
            train_set = vit_neural_temp_dataset(dataset=train_data, label=train_label, limited_6=True, neuron_set=(neuron_set_train, neuron_set))
            test_set = vit_neural_temp_dataset(dataset=test_data, label=test_label, limited_6=True, neuron_set=(neuron_set_train, neuron_set))
        else:
            train_set = vit_neural_dataset(dataset=train_data, label=train_label)
            test_set = vit_neural_dataset(dataset=test_data, label=test_label)

        return train_set, test_set

    def single_animal_loader(self, animal, day):
        """main dataset for xps"""
        train_set, test_set = self.single_animal_dataset(animal, day)

        train_loader = DataLoader(train_set, batch_size=self.batch_size//self.time_select, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size//self.time_select)

        return train_loader, test_loader

    def single_animal_limited_dynamic_loader(self, animal, day):
        """limited (2) dynamic points"""

        train_set, test_set = self.single_animal_dataset(animal, day, type='limited')
        # print("** train set size **", )

        train_loader = DataLoader(train_set, batch_size=self.batch_size // self.time_select, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size // self.time_select)

        return train_loader, test_loader

    def single_animal_limited_6_dynamic_loader(self, animal, day):
        """limited (2) dynamic points"""

        train_set, test_set = self.single_animal_dataset(animal, day, type='limited_6')

        train_loader = DataLoader(train_set, batch_size=self.batch_size // self.time_select, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size // self.time_select)

        return train_loader, test_loader

    def single_animal_neuron_transfer_2_loader(self, animal, day, set=True):
        train_set, test_set = self.single_animal_dataset(animal, day, type='limited', neuron_set=80, neuron_set_train=set)
        #train_set_neuron, test_set_neuron = self.single_animal_dataset(animal, day, type='limited', neuron_set=80,
        #                                                 neuron_set_train=False)

        train_loader = DataLoader(train_set, batch_size=self.batch_size // self.time_select, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size // self.time_select)

        #train_loader_neuron = DataLoader(train_set_neuron, batch_size=self.batch_size // self.time_select, shuffle=True)
        #test_loader_neuron = DataLoader(test_set_neuron, batch_size=self.batch_size // self.time_select)

        return train_loader, test_loader


    def single_animal_direction_loader(self, animal, day, train_dir, test_dir):
        train_set, test_set = self.single_animal_dataset(animal, day)

        train_dirc_set = direction_dataset(train_set, test_set, train_dir)
        test_dirc_set_trainsplit = direction_dataset(train_set, None, test_dir, comb=False)
        test_dirc_set_testsplit = direction_dataset(test_set, None, test_dir, comb=False)

        train_loader = DataLoader(train_dirc_set, batch_size=self.batch_size//self.time_select, shuffle=True)
        test_loader_trainsplit = DataLoader(test_dirc_set_trainsplit, batch_size=self.batch_size//self.time_select)
        test_loader_testsplit = DataLoader(test_dirc_set_testsplit, batch_size=self.batch_size // self.time_select)

        return train_loader, {"test_trainsplit": test_loader_trainsplit,
                              "test_testsplit": test_loader_testsplit}

    def single_animal_direction_limited_dynamic_loader(self, animal, day, train_dir, test_dir):
        train_set, test_set = self.single_animal_dataset(animal, day, type='limited')

        train_dirc_set = direction_dataset(train_set, test_set, train_dir)
        test_dirc_set_trainsplit = direction_dataset(train_set, None, test_dir, comb=False, test_limited=True)
        test_dirc_set_testsplit = direction_dataset(test_set, None, test_dir, comb=False, test_limited=True)

        train_loader = DataLoader(train_dirc_set, batch_size=self.batch_size//self.time_select, shuffle=True)
        test_loader_trainsplit = DataLoader(test_dirc_set_trainsplit, batch_size=self.batch_size//self.time_select)
        test_loader_testsplit = DataLoader(test_dirc_set_testsplit, batch_size=self.batch_size // self.time_select)

        return train_loader, {"test_trainsplit": test_loader_trainsplit,
                              "test_testsplit": test_loader_testsplit}

    def single_animal_time_limited_dynamic_loader(self, animal, day):
        train_set, test_set = self.single_animal_dataset(animal, day)

        train_dirc_set = time_dataset(train_set, test_set)
        #test_dirc_set_trainsplit = time_dataset(train_set, None, train=False, comb=False, test_limited=True)
        #test_dirc_set_testsplit = time_dataset(test_set, None, train=False, comb=False, test_limited=True)
        test_dirc_set = time_dataset(train_set, test_set, train=False)

        train_loader = DataLoader(train_dirc_set, batch_size=self.batch_size*2//self.time_select, shuffle=True)
        #test_loader_trainsplit = DataLoader(test_dirc_set_trainsplit, batch_size=self.batch_size*2// self.time_select)
        #test_loader_testsplit = DataLoader(test_dirc_set_testsplit, batch_size=self.batch_size*2// self.time_select)
        test_loader = DataLoader(test_dirc_set, batch_size=self.batch_size*2//self.time_select)

        return train_loader, test_loader

    def multiple_animal_loader(self, animals, days):
        assert len(animals) == len(days)

        train_sets = []
        test_sets = []
        for i in range(len(animals)):
            train_set, test_set = self.single_animal_dataset(animals[i], days[i])
            train_sets.append(train_set)
            test_sets.append(test_set)

        TYPE = 'cat'
        if TYPE == 'list':
            train_multiset = multiple_datasets_list(train_sets)
            test_multiset = multiple_datasets_list(test_sets)

            train_loader = DataLoader(train_multiset, batch_size=self.batch_size//self.time_select,
                                      shuffle=True, collate_fn=self.list_collate)
            test_loader = DataLoader(test_multiset, batch_size=self.batch_size//self.time_select,
                                     collate_fn=self.list_collate)
        elif TYPE == 'cat':
            """basically, it returns a tuple with 4 animals
            where data[0] gives data, label = data[0] for animal 1"""
            train_multiset = multiple_datasets_cat(train_sets)
            test_multiset = multiple_datasets_cat(test_sets)

            train_loader = DataLoader(train_multiset, batch_size=self.batch_size//(self.time_select*4), shuffle=True)
            test_loader = DataLoader(test_multiset, batch_size=self.batch_size//(self.time_select*4))
        else:
            raise NotImplementedError("a better idea??")

        return train_loader, test_loader

    @staticmethod
    def list_collate(batch):
        # print(batch)
        data = [item[0] for item in batch]
        target = [torch.LongTensor(item[1]) for item in batch]
        # target = torch.LongTensor(target)
        return [data, target]


    def multi_resolution_loader(self, animal, day):
        raise NotImplementedError('god someone help me write this pls?')
