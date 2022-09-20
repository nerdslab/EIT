import os
import numpy as np
import pickle
from tqdm import tqdm

from einops import rearrange, repeat
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from neural_kits.utils import set_random_seeds
from neural_kits.neural_models import Neural_ViT_T, Neural_ViT_S, Neural_ViT_Benchmark
from neural_kits.neural_trainer import vit_neural_learner, vit_neural_learner_trans

from synthetic_utils import two_in_four, perm_expert_4

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class Two_body_Dataset_whole(Dataset):
    def __init__(self, path='datasets/synthetic_twobody/test8.npz',
                 train_ratio=0.8,
                 train=True,
                 label_type='both',
                 plot=False,
                 ):
        dat = np.load(path)
        len_data = dat['data'].shape[0]
        train_split = round(train_ratio*len_data)

        self.data = []
        self.label = []

        self.exp = perm_expert_4()
        self.perm = self.exp.all_possible_perm
        if label_type == 'type':
            self.label_corres = self.exp.label_type
        elif label_type == 'connection':
            self.label_corres = self.exp.label_connection
        elif label_type == 'both':
            self.label_corres = self.exp.label_both

        if train:
            for i in range(24):
                self.data.append(torch.Tensor(dat['data'][:train_split, self.perm[i], 1:3, :]))
                self.label.append(self.label_corres[i] * torch.ones(dat['data'][:train_split].shape[0],
                                                                    dat['data'][:train_split].shape[-1]))

        else:
            for i in range(24):
                self.data.append(torch.Tensor(dat['data'][train_split:, self.perm[i], 1:3, :]))
                self.label.append(self.label_corres[i] * torch.ones(dat['data'][train_split:].shape[0],
                                                                    dat['data'][train_split:].shape[-1]))

        if plot:
            self.data = self.data[0]
            self.label = self.label[0]
        else:
            self.data = torch.cat(self.data)
            self.label = (torch.cat(self.label)).long()

    def __getitem__(self, index):
        # data with shape [trial, 6, 2, timestamp], label is [trial, timestamp]
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


class Synthetic_ViT_T(Neural_ViT_T):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.single_embed = nn.Linear(2, kwargs['single_dim'])

    def get_latent_t(self, img):
        b, n, d, t = img.shape
        img = rearrange(img, 'b n d t -> (b n) t d')

        trans_x = self.single_embed(img)  # [(batch, neuron), time, dim]
        trans_x, t_weights = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]

        trans_x = rearrange(trans_x, '(b n) t d -> b n t d', b=b)  # [batch, neuron, (time, dim)]
        small_trans_x = self.mlp_head_bottolneck(trans_x)
        return trans_x, small_trans_x


class Synthetic_ViT_S(Neural_ViT_S):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, img):
        b, n, d, t = img.shape

        trans_x, small_trans_x = self.MT.get_latent_t(img)

        assert self.type == 'cat'
        trans_x = rearrange(trans_x, 'b n t d -> (b t) n d')

        trans_x_cp = trans_x.clone().detach()

        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b*t)  # [(b t) n ed]

        x = trans_x + embed_x
        x = self.dropout(x)
        x, weights = self.S_transformer(x)  # [(b t) n ed+td]

        store_x = x.clone().detach()
        x = self.bottolneck(x)  # [(b t) n 1]
        store_btnk = x.clone().detach()

        x = rearrange(x, 'b n d -> b (n d)')  # 4*4? --> matrix of 16
        # x = self.weights_norm(x)
        x = self.mlp_head(x)

        return x, {"weights": weights,
                   "trans_x": trans_x_cp,
                   "store_x": store_x,
                   "bottleneck_x": store_btnk,
                   }


class Synthetic_NDT(Neural_ViT_Benchmark):
    def __init__(self, time=10, **kwargs):
        super().__init__(**kwargs)

        self.NDT_linear = nn.Linear(kwargs['neuron']*2, kwargs['neuron']*2)
        self.NDT_temp_embed = nn.Parameter(torch.randn(1, time, kwargs['neuron']*2))

    def forward(self, img):
        b, n, d, t = img.shape

        img = rearrange(img, 'b n d t -> b t (n d)')

        x = self.NDT_linear(img)  # b t dim
        temp_token = repeat(self.NDT_temp_embed, '() t d -> b t d', b=b)
        x = x + temp_token

        x, weights = self.NDT_transformer(x)  # [b t dim]
        x = self.NDT_back(x)
        x = self.NDT_cls(x)  # [b t cls]

        return rearrange(x, 'b t cls -> (b t) cls'), {'weights': weights}


def main(label_type='connection'):
    BS = 64
    neuron_dim = 16
    t_epoch = 500
    LR = 1e-4

    dataset_train = Two_body_Dataset_whole(train=True, label_type=label_type)
    dataset_test = Two_body_Dataset_whole(train=False, label_type=label_type)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_train))
    print(x.shape, label.shape)

    v = Synthetic_ViT_T(
        num_classes=CLS,
        single_dim=neuron_dim,
        depth=1,
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
        depth=1,
        heads=6,
        offset=True,
        ff=False,
    ).cuda()

    # progress recording
    TB_LOG_NAME = "SYN_twobody{}".format(label_type)
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


def benchmark(label_type='connection'):
    BS = 64
    t_epoch = 500
    LR = 5e-4  # needs higher than our model to learn better

    dataset_train = Two_body_Dataset_whole(train=True, label_type=label_type)
    dataset_test = Two_body_Dataset_whole(train=False, label_type=label_type)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_train))
    print(x.shape, label.shape)

    s = Synthetic_NDT(
        num_classes=CLS,
        depth=2,
        heads=6,
        neuron=x.shape[1],
        dropout=0.0,  # change number as needed, 0.0 is slightly worse in general
    ).cuda()

    # progress recording
    TB_LOG_NAME = "SYN_twobody_bm{}".format(label_type)
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


def attention(label_type):
    BS = 512
    neuron_dim = 16

    dataset_train = Two_body_Dataset_whole(train=True, label_type=label_type, plot=True)
    dataset_test = Two_body_Dataset_whole(train=False, label_type=label_type, plot=True)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_train))
    print(x.shape, label.shape)

    v = Synthetic_ViT_T(
        num_classes=CLS,
        single_dim=neuron_dim,
        depth=1,
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
        depth=1,
        heads=6,
        offset=True,
        ff=False,
    ).cuda()

    s.load_state_dict(torch.load('ckpt_syn_bm/SYN_twobody{}/vit_epoch299.pt'.format(label_type)))

    T_net = s.MT
    s.eval()
    T_net.eval()

    with torch.no_grad():

        cls, weights = s(x.cuda())
        label = rearrange(label, 'b t -> (b t)')
        print(dataset_train.perm[1])

        label0_mask = label == 0
        print('mask shape', label0_mask.shape)

        print(len(weights["weights"]), weights["weights"][0].shape) # 2 layers, with 3200 6 6 6

        attns = torch.mean(torch.stack(weights["weights"], dim=1)[label0_mask], dim=0).detach().cpu()  # 2, 6, 6, 6
        attns = torch.mean(attns, dim=1)  # Average the weights across all heads. [2 6 6]]

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        # print(attns[0], '\n', attns[1])

        residual_att = torch.eye(attns.size(1))
        aug_att_mat = attns + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        # aug_att_mat = attns

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        # this is the latents beforehand
        latents_bf, _ = T_net.get_latent_t(x.cuda())
        latents_bf = rearrange(latents_bf, 'b n t d -> (b t) n d')

        total = 200

        pca_bf = PCA(2)
        latents_bf = rearrange(latents_bf, 'b n d -> (b n) d')
        latents_bf = pca_bf.fit_transform(latents_bf.detach().cpu())
        latents_bf = rearrange(latents_bf, '(b n) d -> b n d', n=4)
        latents0_bf = latents_bf[label0_mask]

        id = 1
        traj = [[latents0_bf[id, 0, 0], latents0_bf[id, 1, 0], latents0_bf[id, 2, 0], latents0_bf[id, 3, 0]],
                [latents0_bf[id, 0, 1], latents0_bf[id, 1, 1], latents0_bf[id, 2, 1], latents0_bf[id, 3, 1]]]
        axes[0].plot(traj[0][0:2], traj[1][0:2], c='k')
        axes[0].plot(traj[0][2:], traj[1][2:], c='k')

        axes[0].scatter(latents0_bf[:total, 0, 0], latents0_bf[:total, 0, 1], label='body 0')
        axes[0].scatter(latents0_bf[:total, 1, 0], latents0_bf[:total, 1, 1], label='body 1')
        axes[0].scatter(latents0_bf[:total, 2, 0], latents0_bf[:total, 2, 1], label='body 2')
        axes[0].scatter(latents0_bf[:total, 3, 0], latents0_bf[:total, 3, 1], label='body 3')

        _, latents_af = s(x.cuda())
        latents_af = latents_af['store_x']

        pca_af = PCA(2)
        latents_af = rearrange(latents_af, 'b n d -> (b n) d')
        latents_af = pca_af.fit_transform(latents_af.detach().cpu())
        latents_af = rearrange(latents_af, '(b n) d -> b n d', n=4)
        latents0_af = latents_af[label0_mask]

        id = 1
        traj = [[latents0_af[id, 0, 0], latents0_af[id, 1, 0], latents0_af[id, 2, 0], latents0_af[id, 3, 0]],
                [latents0_af[id, 0, 1], latents0_af[id, 1, 1], latents0_af[id, 2, 1], latents0_af[id, 3, 1]]]
        axes[1].plot(traj[0][0:2], traj[1][0:2], c='k')
        axes[1].plot(traj[0][2:], traj[1][2:], c='k')

        axes[1].scatter(latents0_af[:total, 0, 0], latents0_af[:total, 0, 1], label='body 0')
        axes[1].scatter(latents0_af[:total, 1, 0], latents0_af[:total, 1, 1], label='body 1')
        axes[1].scatter(latents0_af[:total, 2, 0], latents0_af[:total, 2, 1], label='body 2')
        axes[1].scatter(latents0_af[:total, 3, 0], latents0_af[:total, 3, 1], label='body 3')

        plt.legend()
        plt.show()
        #plt.savefig('2body-both-latent-0516.eps')


def visualize_OT(label_type):
    BS = 512
    neuron_dim = 16

    dataset_train = Two_body_Dataset_whole(train=True, label_type=label_type, plot=True)
    dataset_test = Two_body_Dataset_whole(train=False, label_type=label_type, plot=True)

    loader_train = DataLoader(dataset_train, batch_size=BS, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=BS)

    x, label = next(iter(loader_train))
    print(x.shape, label.shape)

    v = Synthetic_ViT_T(
        num_classes=CLS,
        single_dim=neuron_dim,
        depth=1,
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
        depth=1,
        heads=6,
        offset=True,
        ff=False,
    ).cuda()

    # s.load_state_dict(torch.load('ckpt_neural/SYN_twobody/vit_epoch499.pt'))
    s.load_state_dict(torch.load('ckpt_syn_bm/SYN_twobodyboth/vit_epoch399.pt'))

    T_net = s.MT
    s.eval()
    T_net.eval()

    with torch.no_grad():

        cls, weights = s(x.cuda())
        # cls = rearrange(cls, '(b t) d -> b t d', b=BS)
        # print(label[:, 0], cls[:, 0, :]) # santity check of labels

        label = rearrange(label, 'b t -> (b t)')
        print(dataset_train.perm[1])

        label0_mask = label == 0
        print('mask shape', label0_mask.shape)

        print(len(weights["weights"]), weights["weights"][0].shape) # 2 layers, with 3200 6 6 6

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        # this is the latents beforehand
        latents_bf, _ = T_net.get_latent_t(x.cuda())
        latents_bf = rearrange(latents_bf, 'b n t d -> (b t) n d')

        _, latents_af = s(x.cuda())
        latents_af = latents_af['store_x']

        print(latents_bf.shape, label.shape, latents_af.shape) # torch.Size([4000, 4, 16]), torch.Size([4000])
        from ot_distance_funcs import compute_OT

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        # matrix = torch.zeros(4, 4)
        matrix = torch.Tensor(compute_OT(latents_bf.cpu().detach(), latents_bf.cpu().detach()))

        matrix[0, 0] = -1
        matrix[1, 1] = -1
        matrix[2, 2] = -1
        matrix[3, 3] = -1

        img = ax[0].imshow(-matrix, cmap='coolwarm', vmin=-25, vmax=1)

        cbar = fig.colorbar(img, ax=ax)
        #cbar.set_clim(-25, 20)
        #plt.show()
        plt.savefig('2body-distance-coolwarm.eps')


if __name__ == '__main__':


    set_random_seeds(0)

    exp_types = {'connection': 3,
                 'type': 6,
                 'both': 12,
                 }

    key = 'both'

    CLS = exp_types[key]

    #attention(key)
    visualize_OT(key)

