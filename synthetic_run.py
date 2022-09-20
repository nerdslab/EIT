import os
from absl import app
from absl import flags
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader, Dataset

from my_transformers.transforms import *

from neural_kits.neural_dataset import ReachNeuralDataset, vit_neural_dataset
from neural_kits.neural_models import Neural_ViT_bm, Neural_MAE_bm
from neural_kits.neural_trainer import vit_neural_learner, mae_neural_learner
from neural_kits.neural_tasks import angle_linear_clf
from neural_kits.neural_augments import neuron_augmentor, Pepper
from neural_kits.utils import set_random_seeds

import warnings
warnings.filterwarnings("ignore")

from einops import rearrange
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# Dataset

# Model params
flags.DEFINE_integer('en_depth', 4, 'ViT depth')
flags.DEFINE_integer('en_heads', 4, 'ViT heads')
flags.DEFINE_integer('ps', 8, 'patch size')  # 8*8 image = 64 dim
flags.DEFINE_integer('en_dim', 80, 'the dim of encoder')
flags.DEFINE_integer('de_dim', 40, 'the dim of decoder')
flags.DEFINE_integer('mlp_dim', 80, 'the dim of viT mlp')
flags.DEFINE_integer('de_depth', 2, 'decoder depth')

flags.DEFINE_float('masking_ratio', 1/2, 'how many patches of neurons to mask')

flags.DEFINE_integer('subset_depth', 1, 'depth of subset transformer')
flags.DEFINE_integer('subset_heads', 4, 'heads of subset transformer')
flags.DEFINE_integer('subset_mlp', 8, 'mlp dim of subset transformer')
flags.DEFINE_string('subset_type', 'add', 'subset transformer type, cat or add or set')
flags.DEFINE_integer('subset_transdim', 16, 'transformer dim of subset transformer')

# Augmentations

# Model training
flags.DEFINE_integer('t_epoch', 300, 'Total epochs to consider')
flags.DEFINE_float('lr', 0.0001, 'Learning rate, 0.0001 good for MSE loss on 256 size imgs')
flags.DEFINE_integer('bs', 2048, 'Batch size')  # 128 default for synthetic
flags.DEFINE_integer('seed', 0, 'random seed across everything')

# logging
flags.DEFINE_string('MAEorVIT', 'VIT', 'MAE, VIT, or both')
flags.DEFINE_string('version', 'our-test', 'Training version, logfile name')
flags.DEFINE_integer('save_epoch', 200, 'For every xx epochs, the model ckpt is saved')

class synthetic_dataset(Dataset):
    def __init__(self, dat, perm=None, rotate=True, special_pattern=True):
        self.img = torch.Tensor(dat['img'])  # [T, 8*8]
        self.label = torch.LongTensor(dat['label'])

        if perm is not None:
            self.img[:, perm] = self.img.clone()

        if rotate:
            self.img = rearrange(self.img, 'b (w h) -> b w h', h=8, w=8)
            self.img = torch.transpose(self.img, 1, 2)
            self.img = rearrange(self.img, 'b w h -> b (w h)')

        if special_pattern:
            SP_pattern = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                       [14, 15, 8, 9, 10, 11, 12, 13],
                                       [20, 21, 22, 23, 16, 17, 18, 19],
                                       [26, 27, 28, 29, 30, 31, 24, 25]])
            SP_pattern = torch.cat([SP_pattern, SP_pattern+32])
            SP_pattern = rearrange(SP_pattern, 'h w -> (h w)')
            self.img[:, SP_pattern] = self.img.clone()

    def __getitem__(self, index):
        return self.img[index], self.label[index]

    def __len__(self):
        return self.img.shape[0]


def run(args):
    set_random_seeds(FLAGS.seed)

    data_dict = get_data(args)

    if FLAGS.MAEorVIT == "MAE":
        MAE_neural(args, data_dict)
    elif FLAGS.MAEorVIT == "VIT":
        ViT_neural(args, data_dict, LOAD=False)
    elif FLAGS.MAEorVIT == "both":
        MAE_neural(args, data_dict)
        ViT_neural(args, data_dict, LOAD=True)
    elif FLAGS.MAEorVIT == 'v':
        Visual_neuron_complex(args, data_dict)


def get_data(args):

    DATA_PATH = 'datasets/synthetic_sq'
    train_dat = np.load(os.path.join(DATA_PATH, 'train.npz'))
    test_dat = np.load(os.path.join(DATA_PATH, 'test.npz'))

    if PERMUTE:
        neuron_perm = torch.randperm(8 * 8)
    else:
        neuron_perm = None

    train_loader = DataLoader(synthetic_dataset(train_dat, perm=neuron_perm), batch_size=FLAGS.bs, shuffle=True)
    test_loader = DataLoader(synthetic_dataset(test_dat, perm=neuron_perm), batch_size=FLAGS.bs)

    data, label = next(iter(train_loader))
    print(data.shape, label.shape)

    return {'train_loader': train_loader,
            'test_loader': test_loader,
            'perm': neuron_perm}


def ViT_neural_bm(args, data_dict, LOAD):
    """benchmark ViT synthetic experiment for reference"""

    if LOAD:
        LOAD = "syn_ckpt/MAE-{}/vit_epoch199.pt".format(FLAGS.version)

    augmentor = None

    v = Neural_ViT_bm(
        neuron=64,
        patch_size=FLAGS.ps,
        num_classes=4,
        dim=FLAGS.en_dim,  # 8*8*3 = 192
        depth=FLAGS.en_depth,
        heads=FLAGS.en_heads,
        mlp_dim=FLAGS.mlp_dim,  # 32*32*3
        patch2token='normal',
        pool='cls',
    ).cuda()

    if LOAD is not False:
        v.load_state_dict(torch.load(LOAD))

    # progress recording
    TB_LOG_NAME = "VIT-{}".format(FLAGS.version)
    if not os.path.exists("syn_ckpt/{}".format(TB_LOG_NAME)):
        os.makedirs("syn_ckpt/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_syn", name=TB_LOG_NAME)

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = data_dict['train_loader']
    add['test_loader'] = data_dict['test_loader']
    add['en_dim'] = FLAGS.en_dim
    add['save_dict'] = 'syn_ckpt'

    learner = vit_neural_learner(
        vit=v,
        augmentor=augmentor,
        LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, data_dict['train_loader'])


def ViT_neural(args, data_dict, LOAD):

    if LOAD:
        LOAD = "syn_ckpt/MAE-{}/vit_epoch199.pt".format(FLAGS.version)

    augmentor = None

    subset_params = {'subset_depth': FLAGS.subset_depth,  # 1
                     'subset_heads': 6,  # 4
                     'subset_mlp': 16,  # 8 as default -- higher tends to overfit the model
                     'subset_transdim': 16,  # 16 is better than 8
                     'subset_embeddim': 16}

    v = Neural_ViT_bm(
        neuron=64,
        patch_size=FLAGS.ps,
        num_classes=4,
        dim=FLAGS.en_dim,  # 8*8*3 = 192
        depth=FLAGS.en_depth,
        heads=FLAGS.en_heads,
        mlp_dim=FLAGS.mlp_dim,  # 32*32*3
        patch2token='transid',
        pool='cls',
        subset_params=subset_params,
    ).cuda()

    if LOAD is not False:
        v.load_state_dict(torch.load(LOAD))

    # progress recording
    TB_LOG_NAME = "VIT-{}".format(FLAGS.version)
    if not os.path.exists("syn_ckpt/{}".format(TB_LOG_NAME)):
        os.makedirs("syn_ckpt/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_syn", name=TB_LOG_NAME)

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = data_dict['train_loader']
    add['test_loader'] = data_dict['test_loader']
    add['en_dim'] = FLAGS.en_dim
    add['save_dict'] = 'syn_ckpt'

    learner = vit_neural_learner(
        vit=v,
        augmentor=augmentor,
        LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, data_dict['train_loader'])


def MAE_neural(args, data_dict):

    subset_params = {'subset_depth': FLAGS.subset_depth,
                     'subset_heads': FLAGS.subset_heads,
                     'subset_mlp': FLAGS.subset_mlp,
                     'subset_type': FLAGS.subset_type,
                     'subset_transdim': FLAGS.subset_transdim,
                     'patch_dim': FLAGS.ps,
                     'num_patches': 8,
                     'dim': FLAGS.en_dim,
                     }

    masking_ratio = FLAGS.masking_ratio

    v = Neural_ViT_bm(
        neuron=160,
        patch_size=FLAGS.ps,
        num_classes=8,
        dim=FLAGS.en_dim,  # 8*8*3 = 192
        depth=FLAGS.en_depth,
        heads=FLAGS.en_heads,
        mlp_dim=FLAGS.mlp_dim,  # 32*32*3
        subset_params = subset_params,
        masking_ratio=masking_ratio,
        patch2token='normal',
    ).cuda()

    mae = Neural_MAE_bm(
        encoder=v,
        masking_ratio=masking_ratio,  # the paper recommended 75% masked patches
        decoder_dim=FLAGS.de_dim,  # paper showed good results with just 512
        decoder_depth=FLAGS.de_depth,  # anywhere from 1 to 8
    ).cuda()

    TB_LOG_NAME = "MAE{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
    if not os.path.exists("neural_ckpt/{}".format(TB_LOG_NAME)):
        os.makedirs("neural_ckpt/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural_clean", name=TB_LOG_NAME)

    if FLAGS.Aug_MAEorVIT == "MAE" or FLAGS.Aug_MAEorVIT == "both":
        augmentor = neuron_augmentor()
        augmentor.aug_list.append(Pepper(p=FLAGS.pepper_p, sigma=FLAGS.pepper_sigma, apply_p=1.))
    else:
        augmentor = None

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = train_loader
    add['test_loader'] = test_loader
    add['en_dim'] = FLAGS.en_dim
    add['masking_ratio'] = masking_ratio

    learner = mae_neural_learner(
        vit=v,
        mae=mae,
        augmentor=augmentor,
        LR=FLAGS.lr,
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
        SAVE=50,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1, max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_loader)


def Visual_neuron(args, data_dict):

    #LOAD = 'neural_ckpt/VITmihi1-test_attention/vit_epoch399.pt'
    LOAD = 'syn_ckpt/VIT-test3/vit_epoch199.pt'

    subset_params = {'subset_depth': FLAGS.subset_depth,
                     'subset_heads': FLAGS.subset_heads,
                     'subset_mlp': FLAGS.subset_mlp,
                     'subset_type': FLAGS.subset_type,
                     'subset_transdim': FLAGS.subset_transdim,
                     'patch_dim': FLAGS.ps,
                     'num_patches': 8,
                     'dim': FLAGS.en_dim,
                     }

    v = Neural_ViT_bm(
        neuron=64,
        patch_size=FLAGS.ps,
        num_classes=4,
        dim=FLAGS.en_dim,  # 8*8*3 = 192
        depth=FLAGS.en_depth,
        heads=FLAGS.en_heads,
        mlp_dim=FLAGS.mlp_dim,  # 32*32*3
        subset_params=subset_params,
        patch2token='normal'
    ).cuda()
    v.load_state_dict(torch.load(os.path.join(LOAD)))

    data, label = next(iter(data_dict['train_loader']))
    print("data?", data.shape)

    show_data = False
    if show_data:
        fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(8, 2))
        for data_i in range(8):
            axes[data_i].title.set_text('L {}'.format(label[data_i]))
            data_example = rearrange(data[data_i], '(h w) -> h w', h=8)
            axes[data_i].imshow(data_example)
        plt.show()

    v.eval()

    with torch.no_grad():
        _, ori_attns = v(data.cuda())

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8*2, 4*2))
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    y_plts = []
    x_plts = []
    labels = []
    ori_data = []

    for data_i in range(FLAGS.bs):  # batch size
        ori_data_i = data[data_i]
        ori_data.append(ori_data_i)
        label_i = label[data_i]
        labels.append(label_i)

        attns = torch.stack(ori_attns["weights"], dim=1)[data_i].detach().cpu()  # for first img [6, 6, 8, 8]
        attns = torch.mean(attns, dim=1)  # Average the weights across all heads. [6, 8, 8]
        aug_att_mat = attns

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]
        v = torch.mean(v, dim=0)[1:]  # pack weights, shape like [8]
        # v = v[0, 1:]

        y_plts.append(v)

    y_plts = torch.stack(y_plts, dim=0)
    labels = torch.stack(labels, dim=0)

    for i in range(4):  # different labels
        mask = labels == i
        y_plts_i = torch.mean(y_plts[mask], dim=0) - torch.mean(y_plts, dim=0)
        axes[1].plot(np.arange(8), y_plts_i, c=cmap[i], label=i)

    plt.legend()
    plt.show()

    '''
    individual_mean = torch.mean(ori_attns["individual_weights"][0]).detach().cpu()
    patch_mean = torch.mean(torch.stack(ori_attns["weights"], dim=1)).detach().cpu()

    print(ori_attns["individual_weights"][0].shape, len(ori_attns["individual_weights"]))

    def tensor_2_percent(individual_weights, pn=8):
        """individual_weights is of shape pn n n"""
        scaled_weights = []

        for patch_i in range(pn):
            individual_weights_i = individual_weights[patch_i]  # weights of shape [n, n]
            individual_weights_i = torch.mean(individual_weights_i, dim=0)  # weights of shape [n]
            individual_weights_i[individual_weights_i < 0] = 0.  # set lower bound as 0

            select_weights = torch.zeros(individual_weights_i.shape)
            # individual_weights_i
            top_value, top_index = torch.topk(individual_weights_i, k=3)
            select_weights[top_index] = top_value

            sum_i = torch.sum(select_weights)
            scaled_weights.append(select_weights/sum_i)

        return torch.cat(scaled_weights)

    from einops import rearrange

    for data_i in range(FLAGS.bs):  # batch size
        ori_data_i = data[data_i]
        ori_data.append(ori_data_i)
        label_i = label[data_i]
        labels.append(label_i)

        attns = torch.stack(ori_attns["weights"], dim=1)[data_i].detach().cpu()  # for first img [6, 6, 8, 8]
        attns = torch.mean(attns, dim=1)  # Average the weights across all heads. [6, 8, 8]
        aug_att_mat = attns

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]
        v = torch.mean(v, dim=0)[1:]  # pack weights
        v = v.repeat_interleave(20)  # amount of neurons

        individual_weights = rearrange(ori_attns["individual_weights"][0], '(b pn) h n1 n2 -> b pn h n1 n2', pn=8)
        individual_weights = individual_weights.detach().cpu()
        individual_weights = torch.mean(individual_weights, dim=-3)  # b pn n n
        individual_weights = individual_weights[data_i]

        scaled_indi_weights = tensor_2_percent(individual_weights)
        individual_weights_all = torch.Tensor([v[i] * scaled_indi_weights[i] for i in range(160)])

        y_plts.append(individual_weights_all)

        #plt.plot(np.arange(160), individual_weights_all)
        #plt.show()

    y_plts = torch.stack(y_plts, dim=0)
    y_mean = torch.mean(y_plts, dim=0)
    labels = torch.stack(labels, dim=0)
    ori_data = torch.stack(ori_data, dim=0)

    for i in range(8):  # different labels
        mask = labels == i
        y_plts_i = torch.mean(y_plts[mask], dim=0) - y_mean
        axes[0].plot(np.arange(160), y_plts_i, c=cmap[i], label=i)

    ori_data_mean = torch.mean(ori_data, dim=0)

    for i in range(8):  # different labels
        mask = labels == i
        ori_data_plt = torch.mean(ori_data[mask], dim=0) - ori_data_mean
        axes[1].plot(np.arange(160), ori_data_plt, c=cmap[i], label=i)

    plt.legend()
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8 * 2, 4 * 2))
    x_plt = np.arange(8)

    y_plts = []
    x_plts = []
    labels = []
    ori_data = []

    for data_i in range(FLAGS.bs): # batch size

        ori_data_i = data[data_i]
        ori_data.append(ori_data_i)
        label_i = label[data_i]

        attns = torch.stack(ori_attns["weights"], dim=1)[data_i].detach().cpu()  # for first img [6, 6, 8, 8]
        attns = torch.mean(attns, dim=1)  # Average the weights across all heads. [6, 8, 8]

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        #residual_att = torch.eye(attns.size(1))
        #aug_att_mat = attns + residual_att
        #aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        aug_att_mat = attns

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]

        print(label_i, v[0, 1:])

        #axes.plot(x_plt, v[0, 1:], c=cmap[label_i])

        y_plts.append(v[0, 1:])
        labels.append(label_i)

    y_plts = torch.stack(y_plts, dim=0)
    y_mean = torch.mean(y_plts, dim=0)
    labels = torch.stack(labels, dim=0)
    ori_data = torch.stack(ori_data, dim=0)
    for i in range(8):
        mask = labels == i
        y_plts_i = torch.mean(y_plts[mask], dim=0) - y_mean
        axes[0].plot(x_plt, y_plts_i, c=cmap[i], label = i)

    ori_data_mean = torch.mean(ori_data, dim=0)

    for i in range(8):
        mask = labels == i
        ori_data_plt = torch.mean(ori_data[mask], dim=0) - ori_data_mean
        axes[1].plot(np.arange(160), ori_data_plt, c=cmap[i], label = i)

    plt.legend()
    plt.show()
    '''


def Visual_neuron_complex(args, data_dict):

    LOAD = 'syn_ckpt/VIT-our-test/vit_epoch199.pt'

    subset_params = {'subset_depth': FLAGS.subset_depth,  # 1
                     'subset_heads': 6,  # 4
                     'subset_mlp': 16,  # 8 as default -- higher tends to overfit the model
                     'subset_transdim': 16,  # 16 is better than 8
                     'subset_embeddim': 16}

    v = Neural_ViT_bm(
        neuron=64,
        patch_size=FLAGS.ps,
        num_classes=4,
        dim=FLAGS.en_dim,  # 8*8*3 = 192
        depth=FLAGS.en_depth,
        heads=FLAGS.en_heads,
        mlp_dim=FLAGS.mlp_dim,  # 32*32*3
        patch2token='transid',
        pool='cls',
        subset_params=subset_params,
    ).cuda()

    v.load_state_dict(torch.load(LOAD))

    data, label = next(iter(data_dict['train_loader']))
    v.eval()
    with torch.no_grad():
        _, ori_attns = v(data.cuda())

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8 * 2, 4 * 2))
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan']

    y_plts = []
    x_plts = []
    labels = []
    ori_data = []

    #individual_mean = torch.mean(ori_attns["individual_weights"][0]).detach().cpu()
    #patch_mean = torch.mean(torch.stack(ori_attns["weights"], dim=1)).detach().cpu()
    print(ori_attns["individual_weights"][0].shape, len(ori_attns["individual_weights"]))

    def tensor_2_percent(individual_weights, pn=8):
        """individual_weights is of shape pn n n"""
        scaled_weights = []

        for patch_i in range(pn):
            individual_weights_i = individual_weights[patch_i]  # weights of shape [n, n]
            individual_weights_i = torch.mean(individual_weights_i, dim=0)  # weights of shape [n]
            individual_weights_i[individual_weights_i < 0] = 0.  # set lower bound as 0

            select_weights = torch.zeros(individual_weights_i.shape)
            # individual_weights_i
            top_value, top_index = torch.topk(individual_weights_i, k=3)
            select_weights[top_index] = top_value

            sum_i = torch.sum(select_weights)
            scaled_weights.append(select_weights / sum_i)

        return torch.cat(scaled_weights)

    for data_i in range(FLAGS.bs):  # batch size
        ori_data_i = data[data_i]
        ori_data.append(ori_data_i)
        label_i = label[data_i]
        labels.append(label_i)

        attns = torch.stack(ori_attns["weights"], dim=1)[data_i].detach().cpu()  # for first img [6, 6, 8, 8]
        attns = torch.mean(attns, dim=1)  # Average the weights across all heads. [6, 8, 8]
        aug_att_mat = attns

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]
        v = torch.mean(v, dim=0)[1:]  # pack weights
        v = v.repeat_interleave(20)  # amount of neurons

        individual_weights = rearrange(ori_attns["individual_weights"][0], '(b pn) h n1 n2 -> b pn h n1 n2', pn=8)
        individual_weights = individual_weights.detach().cpu()
        individual_weights = torch.mean(individual_weights, dim=-3)  # b pn n n
        individual_weights = individual_weights[data_i]

        scaled_indi_weights = tensor_2_percent(individual_weights)
        individual_weights_all = torch.Tensor([v[i] * scaled_indi_weights[i] for i in range(64)])

        y_plts.append(individual_weights_all)


    y_plts = torch.stack(y_plts, dim=0)
    y_mean = torch.mean(y_plts, dim=0)
    labels = torch.stack(labels, dim=0)
    ori_data = torch.stack(ori_data, dim=0)

    for i in range(4):  # different labels
        mask = labels == i
        y_plts_i = torch.mean(y_plts[mask], dim=0) - torch.mean(y_plts, dim=0)
        axes[0].plot(np.arange(64), y_plts_i, c=cmap[i], label=i)

    for i in range(4):  # different labels
        mask = labels == i
        ori_data_plt = torch.mean(ori_data[mask], dim=0) - torch.mean(ori_data, dim=0)
        axes[1].plot(np.arange(64), ori_data_plt, c=cmap[i], label=i)

    plt.legend()
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8 * 2, 4 * 2))
    x_plt = np.arange(8)

    y_plts = []
    x_plts = []
    labels = []
    ori_data = []

    for data_i in range(FLAGS.bs):  # batch size

        ori_data_i = data[data_i]
        ori_data.append(ori_data_i)
        label_i = label[data_i]

        attns = torch.stack(ori_attns["weights"], dim=1)[data_i].detach().cpu()  # for first img [6, 6, 8, 8]
        attns = torch.mean(attns, dim=1)  # Average the weights across all heads. [6, 8, 8]

        aug_att_mat = attns

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        v = joint_attentions[-1]

        print(label_i, v[0, 1:])

        y_plts.append(v[0, 1:])
        labels.append(label_i)

    y_plts = torch.stack(y_plts, dim=0)
    y_mean = torch.mean(y_plts, dim=0)
    labels = torch.stack(labels, dim=0)
    ori_data = torch.stack(ori_data, dim=0)
    for i in range(8):
        mask = labels == i
        y_plts_i = torch.mean(y_plts[mask], dim=0) - y_mean
        axes[0].plot(x_plt, y_plts_i, c=cmap[i], label=i)

    ori_data_mean = torch.mean(ori_data, dim=0)

    for i in range(4):
        mask = labels == i
        ori_data_plt = torch.mean(ori_data[mask], dim=0) - ori_data_mean
        axes[1].plot(np.arange(64), ori_data_plt, c=cmap[i], label=i)

    plt.legend()
    plt.show()



if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')

    PERMUTE = False
    app.run(run)
