from absl import app
from absl import flags

import os
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from neural_kits.utils import set_random_seeds
from neural_kits.neural_models import Neural_ViT_T, Neural_ViT_S, Neural_ViT_Benchmark
from neural_kits.neural_trainer import vit_neural_learner, vit_neural_learner_trans

from synthetic_utils import perm_expert_6
from synthetic_exp_twobody import Synthetic_ViT_T, Synthetic_ViT_S, Synthetic_NDT

from sklearn.decomposition import PCA
from einops import rearrange, repeat
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

exp_types = {'connection': 10,
             'type': 90,
             'both': 360,
             }

# Training type
flags.DEFINE_string('type', 'type', 'connection or type or both')
flags.DEFINE_string('model', 'v', 'main or bk or v as visual')

class Three_body_Dataset(Dataset):
    def __init__(self,
                 path='datasets/synthetic_threebody/test2.npz',
                 train_ratio=0.8,
                 train=True,
                 label_type='connection'):
        dat = np.load(path)

        len_data = dat['data'].shape[0]
        train_split = round(train_ratio*len_data)

        self.data = []
        self.label = []

        self.exp = perm_expert_6()
        self.perm = self.exp.all_possible_perm

        if label_type == 'type':
            self.label_corres = self.exp.label_type
        elif label_type == 'connection':
            self.label_corres = self.exp.label_connection
        elif label_type == 'both':
            self.label_corres = self.exp.label_both

        if train:
            for i in range(720):
                self.data.append(torch.Tensor(dat['data'][:train_split, self.perm[i], 1:3, :]))
                self.label.append(self.label_corres[i] * torch.ones(dat['data'][:train_split].shape[0],
                                                                    dat['data'][:train_split].shape[-1]))

        else:
            for i in range(720):
                self.data.append(torch.Tensor(dat['data'][train_split:, self.perm[i], 1:3, :]))
                self.label.append(self.label_corres[i] * torch.ones(dat['data'][train_split:].shape[0],
                                                                    dat['data'][train_split:].shape[-1]))



        self.data = torch.cat(self.data)
        self.label = (torch.cat(self.label)).long()

    def __getitem__(self, index):
        # data with shape [trial, 6, 2, timestamp], label is [trial, timestamp]
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

def main(CLS, label_type='both'):
    BS = 64
    neuron_dim = 24
    t_epoch = 10 # 100
    LR = 1e-4

    dataset_train = Three_body_Dataset(train=True, label_type=label_type)
    dataset_test = Three_body_Dataset(train=False, label_type=label_type)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_train))
    print(x.shape, label.shape)

    v = Synthetic_ViT_T(
        num_classes=CLS,
        single_dim=neuron_dim,
        depth=2,
        heads=6,
        neuron=x.shape[1],
    ).cuda()
    # v.load_state_dict(torch.load('ckpt_neural/VIT_Tmihi1-bm/vit_epoch199.pt'))

    s = Synthetic_ViT_S(
        MT=v,
        neuron=x.shape[1],
        num_classes=CLS,
        single_dim=neuron_dim,
        embed_dim=1, # meaningless, placeholder
        depth=2,
        heads=6,
        offset=True,
        ff=False,
    ).cuda()

    # progress recording
    TB_LOG_NAME = "SYN_threebody{}".format(label_type)
    if not os.path.exists("ckpt_syn_bm/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_syn_bm/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_synthetic", name=TB_LOG_NAME)

    # training
    augmentor = None

    add = {}
    add['train_loader'] = loader_train
    add['test_loader'] = loader_test
    add['save_dict'] = 'ckpt_syn_bm'

    learner = vit_neural_learner(
        vit=s,
        augmentor=augmentor,
        LR=LR,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        SAVE=1, # 20
        add=add,
        reshape_label=True,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, loader_train)

def benchmark(CLS, label_type='both'):
    BS = 64
    t_epoch = 100
    LR = 1e-3

    dataset_train = Three_body_Dataset(train=True, label_type=label_type)
    dataset_test = Three_body_Dataset(train=False, label_type=label_type)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_train))
    print(x.shape, label.shape)

    s = Synthetic_NDT(
        time=50,
        num_classes=CLS,
        depth=2,
        heads=6,
        neuron=x.shape[1],
        # dropout=0.0, # change number as needed, 0.0 is much much worse
    ).cuda()

    # progress recording
    TB_LOG_NAME = "SYN_threebody_bm{}".format(label_type)
    if not os.path.exists("ckpt_syn_bm/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_syn_bm/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_synthetic", name=TB_LOG_NAME)

    # training
    augmentor = None

    add = {}
    add['train_loader'] = loader_train
    add['test_loader'] = loader_test
    add['save_dict'] = 'ckpt_syn_bm'

    learner = vit_neural_learner(
        vit=s,
        augmentor=augmentor,
        LR=LR,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        SAVE=50,
        add=add,
        reshape_label=True,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, loader_train)

def run(args):

    key = FLAGS.type
    CLS = exp_types[key]

    if FLAGS.model == 'main':
        main(CLS, key)
    elif FLAGS.model == 'bk':
        benchmark(CLS, key)
    elif FLAGS.model == 'v':
        attention_check(CLS, key)
    else:
        raise NotImplementedError

def attention_check(CLS=10, label_type='connection'):
    BS = 64
    neuron_dim = 24

    dataset_train = Three_body_Dataset(train=True, label_type=label_type)
    dataset_test = Three_body_Dataset(train=False, label_type=label_type)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_test))
    print(x.shape, label.shape)

    v = Synthetic_ViT_T(
        num_classes=CLS,
        single_dim=neuron_dim,
        depth=2,
        heads=6,
        neuron=x.shape[1],
    ).cuda()
    # v.load_state_dict(torch.load('ckpt_neural/VIT_Tmihi1-bm/vit_epoch199.pt'))

    s = Synthetic_ViT_S(
        MT=v,
        neuron=x.shape[1],
        num_classes=CLS,
        single_dim=neuron_dim,
        embed_dim=1,  # meaningless, placeholder
        depth=2,
        heads=6,
        offset=True,
        ff=False,
    ).cuda()

    s.load_state_dict(torch.load('ckpt_syn_bm/SYN_threebody{}/vit_epoch99.pt'.format(label_type)))
    T_net = s.MT
    s.eval()
    T_net.eval()

    with torch.no_grad():
        cls, _ = s(x.cuda())

        label = rearrange(label, 'b t -> (b t)')
        label0_mask = label == 0
        print('mask shape', label0_mask.shape)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        # this is the latents beforehand
        latents_bf, _ = T_net.get_latent_t(x.cuda())
        latents_bf = rearrange(latents_bf, 'b n t d -> (b t) n d')

        print(latents_bf.shape) # torch.Size([3200, 6, 24])

        total = 1000

        pca_bf = PCA(2)
        latents_bf = rearrange(latents_bf, 'b n d -> (b n) d')
        latents_bf = pca_bf.fit_transform(latents_bf.detach().cpu())
        latents_bf = rearrange(latents_bf, '(b n) d -> b n d', n=6)
        latents0_bf = latents_bf[label0_mask]

        axes[0].scatter(latents0_bf[:total, 0, 0], latents0_bf[:total, 0, 1], label='body 0')
        axes[0].scatter(latents0_bf[:total, 1, 0], latents0_bf[:total, 1, 1], label='body 1')
        axes[0].scatter(latents0_bf[:total, 2, 0], latents0_bf[:total, 2, 1], label='body 2')
        axes[0].scatter(latents0_bf[:total, 3, 0], latents0_bf[:total, 3, 1], label='body 3')
        axes[0].scatter(latents0_bf[:total, 4, 0], latents0_bf[:total, 4, 1], label='body 4')
        axes[0].scatter(latents0_bf[:total, 5, 0], latents0_bf[:total, 5, 1], label='body 5')

        _, latents_af = s(x.cuda())
        latents_af = latents_af['store_x']

        pca_af = PCA(2)
        latents_af = rearrange(latents_af, 'b n d -> (b n) d')
        latents_af = pca_af.fit_transform(latents_af.detach().cpu())
        latents_af = rearrange(latents_af, '(b n) d -> b n d', n=6)
        latents0_af = latents_af[label0_mask]

        axes[1].scatter(latents0_af[:total, 0, 0], latents0_af[:total, 0, 1], label='body 0')
        axes[1].scatter(latents0_af[:total, 1, 0], latents0_af[:total, 1, 1], label='body 1')
        axes[1].scatter(latents0_af[:total, 2, 0], latents0_af[:total, 2, 1], label='body 2')
        axes[1].scatter(latents0_af[:total, 3, 0], latents0_af[:total, 3, 1], label='body 3')
        axes[1].scatter(latents0_af[:total, 4, 0], latents0_af[:total, 4, 1], label='body 4')
        axes[1].scatter(latents0_af[:total, 5, 0], latents0_af[:total, 5, 1], label='body 5')

        plt.legend()
        #plt.show()
        plt.savefig('3body-{}-99.eps'.format(label_type))





if __name__ == '__main__':
    set_random_seeds(0)
    # print(len(perm))
    # main()
    # attention()
    # benchmark()
    # main()
    # attention_bm()
    app.run(run)

    # attention_check()
