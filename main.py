"""main script for all kinds of experiments"""

from absl import app
from absl import flags

import os
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch.nn as nn

from my_transformers.transforms import *

from neural_kits.neural_models import Neural_ViT_T, Neural_ViT_S, Neural_ViT_Benchmark
from neural_kits.neural_trainer import vit_neural_learner, vit_neural_learner_dirc, ndt_ssl_neural_learner
from neural_kits.neural_tasks import transfer_mlp
from neural_kits.utils import set_random_seeds, get_animal_data

import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)


FLAGS = flags.FLAGS

# Training Dataset
flags.DEFINE_string('animal', 'mihi', 'chewie or mihi or M or C')
flags.DEFINE_integer('day', 1, '1 or 2')
flags.DEFINE_string('limited', 'yes', 'yes or no or 6, normally yes')

# Transfer Dataset. Load training model on below data
flags.DEFINE_string('animal_trans', 'mihi', 'chewie or mihi or M or C')
flags.DEFINE_integer('day_trans', 1, '1 or 2')

# Model params, model contains two parts, the T_trans and S_trans
flags.DEFINE_integer('T_depth', 2, 'ViT depth')  # default is 2 or 4
flags.DEFINE_integer('T_heads', 6, 'ViT heads')  # default is 6 or 8
flags.DEFINE_integer('neuron_dim', 16, 'ViT single_dim')
flags.DEFINE_integer('activity_dim', 2 * 8, 'ViT embed_dim')
flags.DEFINE_integer('S_depth', 2, 'ViT depth')  # default is 2
flags.DEFINE_integer('S_heads', 6, 'ViT heads')  # default is 6

# Augmentations
flags.DEFINE_string('Aug_MAEorVIT', 'none', 'MAE, VIT, both, or none')
flags.DEFINE_float('pepper_p', 0.2, 'probablity of applying pepper')
flags.DEFINE_float('pepper_sigma', 1.0, 'strength of applying pepper')
flags.DEFINE_float('neuron_dropout', 0.5, 'strength of neuron dropout')

# Model training
flags.DEFINE_integer('t_epoch', 400, 'Total epochs to consider')
flags.DEFINE_float(
    'lr', 0.0001, 'Learning rate, 0.0001 good for MSE loss on 256 size imgs')
flags.DEFINE_integer('bs', 128, 'Batch size')
# set 15 for most experiments
flags.DEFINE_integer('seed', 15, 'random seed across everything')

flags.DEFINE_float(
    'NDT_mask', 1 / 2, 'masking for NDT, 1/6 for limited 6, 1/2 for limited yes')

# logging
flags.DEFINE_string('MAEorVIT', 'VIT_S', 'VIT_T or VIT_S.'
                                         'v for visualization,'
                                         'trans for transfer animal,'
                                         'trans_dir for transfer direction,'
                                         'trans_time for transfer time'
                                         'NDT for NDT benchmark'
                                         'ssl for self-supervised animal transfer')
flags.DEFINE_string('version', 'repeat-2', 'Training version, logfile name')
flags.DEFINE_integer(
    'save_epoch', 200, 'For every xx epochs, the model ckpt is saved')


neuron_amount = {'chewie1': 163,
                 'chewie2': 148,
                 'mihi1': 163,
                 'mihi2': 152, }


def run(args):
    set_random_seeds(FLAGS.seed)

    data_helper = get_animal_data(
        time_select=8, binning_size=0.1, batch_size=FLAGS.bs)
    if FLAGS.limited == 'no':
        train_temp_loader, test_temp_loader = data_helper.single_animal_loader(
            animal=FLAGS.animal, day=FLAGS.day)
    elif FLAGS.limited == 'yes':
        train_temp_loader, test_temp_loader = data_helper.single_animal_limited_dynamic_loader(
            animal=FLAGS.animal, day=FLAGS.day)
    elif FLAGS.limited == '6':
        train_temp_loader, test_temp_loader = data_helper.single_animal_limited_6_dynamic_loader(animal=FLAGS.animal,
                                                                                                 day=FLAGS.day)
    elif FLAGS.limited == 'neuron':
        train_temp_loader, test_temp_loader = data_helper.single_animal_neuron_transfer_2_loader(animal=FLAGS.animal,
                                                                                                 day=FLAGS.day, set=True)

    print(train_temp_loader.__len__())

    if FLAGS.MAEorVIT == 'VIT_T':
        print("AttN is all you need -- main T training")
        ViT_neural_T(args, train_temp_loader, test_temp_loader)

    elif FLAGS.MAEorVIT == 'VIT_S':
        print("AttN is all you need -- main S training")
        ViT_neural_S(args, train_temp_loader, test_temp_loader)

    elif FLAGS.MAEorVIT == 'trans':
        print("AttN is all you need -- transfer animal")

        if FLAGS.limited == 'no':
            trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans,
                                                                                     day=FLAGS.day_trans)
        elif FLAGS.limited == 'yes':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)
        elif FLAGS.limited == '6':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_6_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)

        ViT_neural_T_trans(args, trans_train_loader,
                           trans_test_loader, train=False)

    elif FLAGS.MAEorVIT == 'trans_dir':
        print("AttN is all you need -- transfer direction")
        # setting 1
        # trans_train, trans_test = [0, 1, 2, 3], [4, 5, 6, 7]
        # setting 2
        trans_train, trans_test = [0, 1], [2, 3, 4, 5, 6, 7]
        dirc_info = {'train_dirc': trans_train,
                     'test_dirc': trans_test}

        if FLAGS.limited == 'yes':
            train_dirc_loader, test_dict = data_helper.single_animal_direction_limited_dynamic_loader(animal=FLAGS.animal,
                                                                                                      day=FLAGS.day,
                                                                                                      train_dir=trans_train,
                                                                                                      test_dir=trans_test)
        elif FLAGS.limited == 'no':
            train_dirc_loader, test_dict = data_helper.single_animal_direction_loader(animal=FLAGS.animal, day=FLAGS.day,
                                                                                      train_dir=trans_train, test_dir=trans_test)

        ViT_neural_T_trans(args, train_dirc_loader, test_dict,
                           train=False, dirc=True, dirc_info=dirc_info)

    elif FLAGS.MAEorVIT == 'trans_time':
        print("AttN is all you need -- transfer time")
        assert FLAGS.limited == 'yes'
        train_dirc_loader, test_dict = data_helper.single_animal_time_limited_dynamic_loader(animal=FLAGS.animal,
                                                                                             day=FLAGS.day)
        ViT_neural_T_trans_time(args, train_dirc_loader, test_dict)

    elif FLAGS.MAEorVIT == 'NDT':
        print('Benchmarking with NDT')
        NDT_benchmark(args, train_temp_loader, test_temp_loader, ssl=True)

    else:
        raise NotImplementedError


def ViT_neural_T(args, train_loader, test_loader):

    data, label = next(iter(train_loader))

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=data.shape[-1],
        neuron_dropout=FLAGS.neuron_dropout,
    ).cuda()

    # progress recording
    TB_LOG_NAME = "VIT_T{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
    if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

    #augmentor = neuron_augmentor()
    #augmentor.aug_list.append(Pepper(p=0.2, sigma=1, apply_p=1.))
    augmentor = None

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = train_loader
    add['test_loader'] = test_loader

    learner = vit_neural_learner(
        vit=v,
        augmentor=augmentor,
        LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
        reshape_label=True,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_loader)


def ViT_neural_S(args, train_loader, test_loader):

    data, label = next(iter(train_loader))
    print(data.shape)

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=data.shape[-1],
        neuron_dropout=FLAGS.neuron_dropout,
    ).cuda()
    # v.load_state_dict(torch.load('ckpt_neural/VIT_Tmihi1-bm/vit_epoch199.pt'))

    s = Neural_ViT_S(
        MT=v,
        neuron=data.shape[-1],
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        embed_dim=FLAGS.activity_dim,
        depth=FLAGS.S_depth,
        heads=FLAGS.S_heads,
        neuron_dropout=FLAGS.neuron_dropout,
        # type='cls',
    ).cuda()

    print(v)

    # progress recording
    TB_LOG_NAME = "VIT_S{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
    if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

    #augmentor = neuron_augmentor()
    #augmentor.aug_list.append(Pepper(p=0.2, sigma=1, apply_p=1.))
    augmentor = None

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = train_loader
    add['test_loader'] = test_loader
    add['save_t'] = True

    learner = vit_neural_learner(
        vit=s,
        augmentor=augmentor,
        LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
        reshape_label=True,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_loader)


def ViT_neural_T_trans(args, train_loader, test_loader,
                       train, dirc=False, dirc_info=None):

    if dirc:
        cls_train = len(dirc_info['train_dirc'])

        # '''
        v = Neural_ViT_T(
            num_classes=cls_train,
            single_dim=FLAGS.neuron_dim,
            depth=FLAGS.T_depth,
            heads=FLAGS.T_heads,
            neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
        ).cuda()

        s = Neural_ViT_S(
            MT=v,
            neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
            num_classes=cls_train,
            single_dim=FLAGS.neuron_dim,
            embed_dim=FLAGS.activity_dim,
            depth=FLAGS.S_depth,
            heads=FLAGS.S_heads,
            neuron_dropout=FLAGS.neuron_dropout,
        ).cuda()

        # progress recording
        TB_LOG_NAME = "VIT_direction{}{}-{}".format(
            FLAGS.animal, FLAGS.day, FLAGS.version)
        if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
            os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
        logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

        augmentor = None

        # additional things for visualizations etc
        add = {}
        add['test_loader_trainsplit'] = test_loader['test_trainsplit']
        add['test_loader_testsplit'] = test_loader['test_testsplit']
        add['neuron_dropout'] = FLAGS.neuron_dropout
        add['save_t'] = True
        add['test_cls'] = len(dirc_info['test_dirc'])

        learner = vit_neural_learner_dirc(
            vit=s,
            augmentor=augmentor,
            LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
            TB_LOG_NAME=TB_LOG_NAME,
            add=add,
            reshape_label=True,
        )

        # change gpus and distributed backend if you want to just use 1 gpu
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=FLAGS.t_epoch,
            accumulate_grad_batches=1,
            # distributed_backend="ddp",
            logger=logger,
        )

        trainer.fit(learner, train_loader)

    else:
        data, label = next(iter(train_loader))

        v = Neural_ViT_T(
            num_classes=8,
            single_dim=FLAGS.neuron_dim,
            depth=FLAGS.T_depth,
            heads=FLAGS.T_heads,
            neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
        ).cuda()

        LOAD = 'ckpt_neural/VIT_S{}{}-0512-6-store/vit_T_epoch199.pt'.format(
            FLAGS.animal, FLAGS.day)
        v.load_state_dict(torch.load(LOAD))
        v.eval()

        TB_LOG_NAME = "VIT_trans{}{}-to-{}{}-{}".format(
            FLAGS.animal, FLAGS.day, FLAGS.animal_trans, FLAGS.day_trans, FLAGS.version)
        if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
            os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
        logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

        data, label = next(iter(train_loader))
        print('trans test, data shape', data.shape)

        class transfer_MLP(nn.Module):
            def __init__(self, dropout, input, cls=8):
                super().__init__()

                self.linear = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(input, cls)).cuda()

            def forward(self, img):
                return self.linear(img)

        MLP = transfer_MLP(dropout=FLAGS.neuron_dropout,
                           input=data.shape[-1], cls=8).cuda()
        MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-4)

        transfer_mlp(v, MLP, MLP_optim, train_loader,
                     test_loader, logger, total_epoch=600)


def ViT_neural_T_trans_time(args, train_loader, test_loader):

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
    ).cuda()

    s = Neural_ViT_S(
        MT=v,
        neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        embed_dim=FLAGS.activity_dim,
        depth=FLAGS.S_depth,
        heads=FLAGS.S_heads,
        neuron_dropout=FLAGS.neuron_dropout,
    ).cuda()

    # progress recording
    TB_LOG_NAME = "VIT_time{}{}-{}".format(FLAGS.animal,
                                           FLAGS.day, FLAGS.version)
    if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

    augmentor = None

    # additional things for visualizations etc
    add = {}
    #add['test_loader_trainsplit'] = test_loader['test_trainsplit']
    #add['test_loader_testsplit'] = test_loader['test_testsplit']
    add['test_loader'] = test_loader
    add['neuron_dropout'] = FLAGS.neuron_dropout
    add['save_t'] = True
    add['test_cls'] = 8

    learner = vit_neural_learner_dirc(
        vit=s,
        augmentor=augmentor,
        LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
        reshape_label=True,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_loader)


def NDT_benchmark(args, train_loader, test_loader, ssl=False):

    data, label = next(iter(train_loader))

    v = Neural_ViT_Benchmark(
        num_classes=8,
        depth=int(FLAGS.T_depth + FLAGS.S_depth),
        heads=FLAGS.T_heads,
        neuron=data.shape[-1],
        # final_dim=16,
        ssl=ssl,
        ssl_ratio=FLAGS.NDT_mask,
    ).cuda()

    # progress recording
    TB_LOG_NAME = "NDT{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
    if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

    augmentor = None

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = train_loader
    add['test_loader'] = test_loader
    add['neuron_shape'] = data.shape[-1]

    if ssl:
        learner = ndt_ssl_neural_learner(
            vit=v,
            augmentor=augmentor,
            LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
            TB_LOG_NAME=TB_LOG_NAME,
            add=add,
            reshape_label=True,
        )
    else:
        learner = vit_neural_learner(
            vit=v,
            augmentor=augmentor,
            LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
            TB_LOG_NAME=TB_LOG_NAME,
            add=add,
            reshape_label=True,
        )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=FLAGS.t_epoch,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_loader)


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')

    app.run(run)
