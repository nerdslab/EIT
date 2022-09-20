from absl import app
from absl import flags

import os
from tqdm import tqdm
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import torch.nn as nn

from my_transformers.transforms import *

from neural_kits.neural_models import Neural_ViT_T, Neural_ViT_S, Neural_ViT_Benchmark, Neural_ViT_S_SSL, Small_translate
from neural_kits.benchmark_models import Neural_MLP, Neural_GRU, Neural_beta, Neural_swap, \
    Transfer_MLP, Transfer_MLP_new_end, Transfer_NDT, Transfer_NDT_new_end
from neural_kits.neural_trainer import vit_neural_learner, vit_neural_learner_G, \
    vit_neural_learner_trans, vit_neural_learner_dirc, vit_neural_SSL_learner, ndt_ssl_neural_learner
from neural_kits.neural_tasks import transfer_mlp, transfer_bchmk, transfer_bdt_bchmk
from neural_kits.utils import set_random_seeds, get_animal_data
from neural_kits.translate_helper import translate_helper_H

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from einops.layers.torch import Rearrange

import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)


FLAGS = flags.FLAGS

# Training Dataset
flags.DEFINE_string('animal', 'mihi', 'chewie or mihi or M or C')
flags.DEFINE_integer('day', 1, '1 or 2')
flags.DEFINE_string('limited', 'yes', 'yes or no or 6 or neuron, normally yes')

# Transfer Dataset. Load training model on below data
flags.DEFINE_string('transfer', 'test', 'train or test')
flags.DEFINE_string('animal_trans', 'mihi', 'chewie or mihi or M or C')
flags.DEFINE_integer('day_trans', 1, '1 or 2')

# Model params, model contains two parts, the T_trans and S_trans
flags.DEFINE_integer('T_depth', 2, 'ViT depth')  # default is 2 or 4
flags.DEFINE_integer('T_heads', 6, 'ViT heads')  # default is 6 or 8
flags.DEFINE_integer('neuron_dim', 16, 'ViT single_dim')
flags.DEFINE_integer('activity_dim', 2*8, 'ViT embed_dim')
flags.DEFINE_integer('S_depth', 2, 'ViT depth')  # default is 2
flags.DEFINE_integer('S_heads', 6, 'ViT heads')  # default is 6

# Augmentations
flags.DEFINE_string('Aug_MAEorVIT', 'none', 'MAE, VIT, both, or none')
flags.DEFINE_float('pepper_p', 0.2, 'probablity of applying pepper')
flags.DEFINE_float('pepper_sigma', 1.0, 'strength of applying pepper')
flags.DEFINE_float('neuron_dropout', 0.5, 'strength of neuron dropout')

# Model training
flags.DEFINE_integer('t_epoch', 400, 'Total epochs to consider')
flags.DEFINE_float('lr', 0.0001, 'Learning rate, 0.0001 good for MSE loss on 256 size imgs')
flags.DEFINE_integer('bs', 128, 'Batch size')
flags.DEFINE_integer('seed', 15, 'random seed across everything')  # set 15 for most experiments

flags.DEFINE_float('NDT_mask', 1/2, 'masking for NDT, 1/6 for limited 6, 1/2 for limited yes')
# logging
flags.DEFINE_string('MAEorVIT', 'VIT_S', 'VIT_T or VIT_S.'
                                         'v for visualization,'
                                         'trans for transfer animal,'
                                         'trans_dir for transfer direction,'
                                         'trans_time for transfer time'
                                         'NDT for NDT benchmark'
                                         'ssl for self-supervised animal transfer'
                                         'transfer_benchmark for the last table models')
flags.DEFINE_string('version', 'repeat-2', 'Training version, logfile name')
flags.DEFINE_integer('save_epoch', 200, 'For every xx epochs, the model ckpt is saved')


neuron_amount = {'chewie1': 163,
                 'chewie2': 148,
                 'mihi1': 163,
                 'mihi2': 152,}


def run(args):
    set_random_seeds(FLAGS.seed)

    data_helper = get_animal_data(time_select=8, binning_size=0.1, batch_size=FLAGS.bs)
    if FLAGS.limited == 'no':
        train_temp_loader, test_temp_loader = data_helper.single_animal_loader(animal=FLAGS.animal, day=FLAGS.day)
    elif FLAGS.limited == 'yes':
        train_temp_loader, test_temp_loader = data_helper.single_animal_limited_dynamic_loader(animal=FLAGS.animal, day=FLAGS.day)
    elif FLAGS.limited == '6':
        train_temp_loader, test_temp_loader = data_helper.single_animal_limited_6_dynamic_loader(animal=FLAGS.animal,
                                                                                               day=FLAGS.day)
    elif FLAGS.limited == 'neuron':
        train_temp_loader, test_temp_loader = data_helper.single_animal_neuron_transfer_2_loader(animal=FLAGS.animal,
                                                                                                 day=FLAGS.day, set=True)

    print(train_temp_loader.__len__())

    FULL_animal = ['chewie', 'chewie', 'mihi', 'mihi']
    FULL_day = [1, 2, 1, 2]
    # train_full_loader, test_full_loader = data_helper.multiple_animal_loader(FULL_animal, FULL_day)

    # data = next(iter(train_full_loader))
    # print(data[0][0].shape, len(data[0]))
    # for i in data:
        # print(i.shape)
        # torch.Size([8, 163])
        # torch.Size([8, 148])


    if FLAGS.MAEorVIT == 'v':
        print("Visualization")
        Visual_neuron(args, train_temp_loader, test_temp_loader)
    elif FLAGS.MAEorVIT == 'temp':
        raise NotImplementedError

    elif FLAGS.MAEorVIT == 'VIT_T':
        print("AttN is all you need -- main T training")
        ViT_neural_T(args, train_temp_loader, test_temp_loader)
    elif FLAGS.MAEorVIT == 'VIT_S':
        print("AttN is all you need -- main S training")
        ViT_neural_S(args, train_temp_loader, test_temp_loader)

    elif FLAGS.MAEorVIT == 'trans':
        print("AttN is all you need -- transfer animal")
        if FLAGS.transfer == 'train':
            print("this does not actully work, fix it and delete this print")
            ...
        else:
            # trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans, day=FLAGS.day_trans)
            # print(trans_loader.__len__())
            if FLAGS.limited == 'no':
                trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans,
                                                                                         day=FLAGS.day_trans)
            elif FLAGS.limited == 'yes':
                trans_train_loader, trans_test_loader = data_helper.single_animal_limited_dynamic_loader(
                    animal=FLAGS.animal_trans, day=FLAGS.day_trans)
            elif FLAGS.limited == '6':
                trans_train_loader, trans_test_loader = data_helper.single_animal_limited_6_dynamic_loader(
                    animal=FLAGS.animal_trans, day=FLAGS.day_trans)

        ViT_neural_T_trans(args, trans_train_loader, trans_test_loader, train=False)

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

        ViT_neural_T_trans(args, train_dirc_loader, test_dict, train=False, dirc=True, dirc_info=dirc_info)
    elif FLAGS.MAEorVIT == 'trans_time':
        print("AttN is all you need -- transfer time")
        assert FLAGS.limited == 'yes'
        train_dirc_loader, test_dict = data_helper.single_animal_time_limited_dynamic_loader(animal=FLAGS.animal,
                                                                                             day=FLAGS.day)
        ViT_neural_T_trans_time(args, train_dirc_loader, test_dict)


    elif FLAGS.MAEorVIT == 'NDT':
        print('Benchmarking with NDT')
        NDT_benchmark(args, train_temp_loader, test_temp_loader, ssl=True)
    elif FLAGS.MAEorVIT == 'ssl':
        print('SSL Transfer Across Animals')
        if FLAGS.limited == 'no':
            trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans,
                                                                                     day=FLAGS.day_trans)
        elif FLAGS.limited == 'yes':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)
        else:
            raise NotImplementedError
        ViT_neural_T_trans_ssl(args, trans_train_loader, trans_test_loader)
    elif FLAGS.MAEorVIT == 'other':
        other_benchmark(args, train_temp_loader, test_temp_loader)
    elif FLAGS.MAEorVIT == 'transfer_benchmark':
        if FLAGS.limited == 'no':
            trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans,
                                                                                     day=FLAGS.day_trans)
        elif FLAGS.limited == 'yes':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)
        elif FLAGS.limited == '6':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_6_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)

        transfer_benchmark(args,
                           train_A_loader=train_temp_loader,
                           test_A_loader=test_temp_loader,
                           train_B_loader=trans_train_loader,
                           test_B_loader=trans_test_loader,)
    elif FLAGS.MAEorVIT == 'translate':
        if FLAGS.limited == 'no':
            trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans,
                                                                                     day=FLAGS.day_trans)
        elif FLAGS.limited == 'yes':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)
        elif FLAGS.limited == '6':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_6_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)

        translate(args, train_A_loader=train_temp_loader, test_A_loader=test_temp_loader,
                  train_B_loader=trans_train_loader, test_B_loader=trans_test_loader,)

    elif FLAGS.MAEorVIT == 'distance':
        if FLAGS.limited == 'no':
            trans_train_loader, trans_test_loader = data_helper.single_animal_loader(animal=FLAGS.animal_trans,
                                                                                     day=FLAGS.day_trans)
        elif FLAGS.limited == 'yes':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)
        elif FLAGS.limited == '6':
            trans_train_loader, trans_test_loader = data_helper.single_animal_limited_6_dynamic_loader(
                animal=FLAGS.animal_trans, day=FLAGS.day_trans)

        distance(args, train_A_loader=train_temp_loader, test_A_loader=test_temp_loader,
                  train_B_loader=trans_train_loader, test_B_loader=trans_test_loader,)
    else:
        pass

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
        #type='cls',
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



def ViT_neural_T_trans(args, train_loader, test_loader, train, dirc=False, dirc_info=None):

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
        '''
        s = Neural_ViT_T(
            num_classes=cls_train,
            single_dim=FLAGS.neuron_dim,
            depth=FLAGS.T_depth,
            heads=FLAGS.T_heads,
            neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
        ).cuda()
        '''
        '''
        # MLP trans model
        s = Neural_MLP(neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                       time=2, dropout=0.2,
                       latent=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                       cls=cls_train)
        '''
        '''
        # GRU trans model
        s = Neural_GRU(num_classes=cls_train,  # amount of final classification cls
                       neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                       dropout_p=0.2,
                       hidden_size=80,)
        '''
        '''
        # NDT trans model
        s = Neural_ViT_Benchmark(num_classes=cls_train,
                                 depth=int(FLAGS.T_depth+FLAGS.S_depth),
                                 heads=FLAGS.T_heads,
                                 neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                                 dropout=0.2,
                                 final_dim=None,)
        '''

        # progress recording
        TB_LOG_NAME = "VIT_direction{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
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

        '''
        s = Neural_ViT_S(
            MT=v,
            neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
            num_classes=8,
            single_dim=FLAGS.neuron_dim,
            embed_dim=FLAGS.activity_dim,
            depth=FLAGS.S_depth,
            heads=FLAGS.S_heads,
            neuron_dropout=FLAGS.neuron_dropout,
            # type='cls',
        ).cuda()
        '''

        if train:
            raise NotImplementedError

            FULL_animal = ['chewie', 'chewie', 'mihi', 'mihi']
            FULL_day = [1, 2, 1, 2]

            datasets_info = {'animals': FULL_animal,
                             'days': FULL_day,}

            v.bottom_head_within_cls()
            v.train()

            # progress recording
            TB_LOG_NAME = "VIT_trans{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
            if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
                os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
            logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

            augmentor = None

            # additional things for visualizations etc
            add = {}
            add['train_loader'] = train_loader
            add['test_loader'] = test_loader

            learner = vit_neural_learner_trans(
                vit=v,
                datasets_info=datasets_info,
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
            # ckpt_neural/VIT_Smihi1-test-norm/vit_T_epoch199.pt
            # LOAD = 'ckpt_neural/VIT_S{}{}-v1-store/vit_T_epoch199.pt'.format(FLAGS.animal, FLAGS.day)
            # LOAD = 'ckpt_neural/VIT_S{}{}-v1-store-limited/vit_T_epoch199.pt'.format(FLAGS.animal, FLAGS.day)

            # ViT-S path used
            # LOAD = 'ckpt_neural/VIT_S{}{}-160-2-store-lr/vit_T_epoch599.pt'.format(FLAGS.animal, FLAGS.day)
            # ViT-T path used
            #LOAD = 'ckpt_neural/VIT_T{}{}-160-2-store/vit_epoch199.pt'.format(FLAGS.animal, FLAGS.day)

            #LOAD = 'ckpt_neural/VIT_T{}{}-160-6-store/vit_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
            #v.load_state_dict(torch.load(LOAD))
            #v.eval()

            # LOAD = 'ckpt_neural/VIT_S{}{}-0506/vit_T_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
            # LOAD = 'ckpt_neural/VIT_S{}{}-0506/vit_T_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
            LOAD = 'ckpt_neural/VIT_S{}{}-0512-6-store/vit_T_epoch199.pt'.format(FLAGS.animal, FLAGS.day)
            v.load_state_dict(torch.load(LOAD))
            v.eval()

            TB_LOG_NAME = "VIT_trans{}{}-to-{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.animal_trans, FLAGS.day_trans, FLAGS.version)
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

            MLP = transfer_MLP(dropout=FLAGS.neuron_dropout, input=data.shape[-1], cls=8).cuda()
            MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-4)

            transfer_mlp(v, MLP, MLP_optim, train_loader, test_loader, logger, total_epoch=600)

def ViT_neural_T_trans_time(args, train_loader, test_loader):

    '''
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
    '''
    '''
    s = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
    ).cuda()
    '''
    '''
    # MLP trans model
    s = Neural_MLP(neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                   time=2, dropout=0.2,
                   latent=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                   cls=8)
    '''
    #'''
    # GRU trans model
    s = Neural_GRU(num_classes=8,  # amount of final classification cls
                   neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                   dropout_p=0.2,
                   hidden_size=80,)
    #'''
    '''
    # NDT trans model
    s = Neural_ViT_Benchmark(num_classes=8,
                             depth=int(FLAGS.T_depth+FLAGS.S_depth),
                             heads=FLAGS.T_heads,
                             neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
                             dropout=0.2,
                             final_dim=None,)
    '''

    # progress recording
    TB_LOG_NAME = "VIT_time{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
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

def ViT_neural_T_trans_ssl(args, train_loader, test_loader):

    data, label = next(iter(train_loader))

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)],
    ).cuda()
    # LOAD = 'ckpt_neural/VIT_S{}{}-v1-store/vit_T_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
    LOAD = 'ckpt_neural/VIT_S{}{}-16-store/vit_T_epoch599.pt'.format(FLAGS.animal, FLAGS.day)

    v.load_state_dict(torch.load(LOAD))
    v.eval()

    s = Neural_ViT_S_SSL(
        neuron=data.shape[-1],
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        embed_dim=FLAGS.activity_dim,
        depth=FLAGS.S_depth,
        heads=FLAGS.S_heads,
        #neuron_dropout=FLAGS.neuron_dropout,
    ).cuda()

    TB_LOG_NAME = "VIT_SSL_trans{}{}-to-{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.animal_trans, FLAGS.day_trans, FLAGS.version)
    if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

    add = {}
    add['train_loader'] = train_loader
    add['test_loader'] = test_loader
    add['neuron_shape'] = data.shape[-1]

    # train SSL and evaluate with MLP
    learner = vit_neural_SSL_learner(
        v=v,
        s=s,
        augmentor=None,
        LR=FLAGS.lr*100,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        TB_LOG_NAME=TB_LOG_NAME,
        add=add,
        reshape_label=True,
    )

    # change gpus and distributed backend if you want to just use 1 gpu
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=200,
        accumulate_grad_batches=1,
        # distributed_backend="ddp",
        logger=logger,
    )

    trainer.fit(learner, train_loader)

    #MLP = nn.Sequential(nn.Dropout(FLAGS.neuron_dropout),
    #                    nn.Linear(data.shape[-1], 8)).cuda()
    #MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-4)
    #transfer_mlp(v, MLP, MLP_optim, train_loader, test_loader, logger, total_epoch=400)


def NDT_benchmark(args, train_loader, test_loader, ssl=False):

    data, label = next(iter(train_loader))

    v = Neural_ViT_Benchmark(
        num_classes=8,
        depth=int(FLAGS.T_depth+FLAGS.S_depth),
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

def other_benchmark(args, train_loader, test_loader):

    data, label = next(iter(train_loader))

    v = Neural_MLP(neuron=data.shape[-1], time=6, latent=round(data.shape[-1]/6)).cuda()
    #v = Neural_GRU(neuron=data.shape[-1], hidden_size=round(data.shape[-1]/2)).cuda()
    # v = Neural_beta(neuron=data.shape[-1], time=8, beta=0.5).cuda()

    #v = Neural_swap(neuron=data.shape[-1], time=2, beta=0.5, alpha=1,
    #                l_dim=data.shape[-1], hidden_dim=[128]).cuda()

    # progress recording
    TB_LOG_NAME = "other{}{}-{}".format(FLAGS.animal, FLAGS.day, FLAGS.version)
    if not os.path.exists("ckpt_neural/{}".format(TB_LOG_NAME)):
        os.makedirs("ckpt_neural/{}".format(TB_LOG_NAME))
    logger = TensorBoardLogger("runs_neural", name=TB_LOG_NAME)

    augmentor = None

    # additional things for visualizations etc
    add = {}
    add['train_loader'] = train_loader
    add['test_loader'] = test_loader
    add['neuron_dim'] = data.shape[-1]

    learner = vit_neural_learner(
        vit=v,
        augmentor=augmentor,
        LR=FLAGS.lr,  # rotation learning rate, 0.001 works better than 0.0001 on MSE loss
        # lr * 10 for small dim GRU
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

def translate(args, train_A_loader, test_A_loader, train_B_loader, test_B_loader):
    """
    logic: train translate model on latents of A
    then apply it on latents of B
    """
    data, label = next(iter(train_A_loader))
    print(data.shape)

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=data.shape[-1],
        neuron_dropout=FLAGS.neuron_dropout,
    ).cuda()

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

    LOAD = 'ckpt_neural/VIT_S{}{}-0506/vit_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
    s.load_state_dict(torch.load(LOAD))

    # create and store latents of T and S on A
    def gather_latents(model, dataloader):
        latents_all = []
        label_all = []

        model.eval()
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.cuda(), label.cuda()

                latents = model.translate_l(data)
                latents_all.append(latents)
                label_all.append(label)

        latents_all = torch.cat(latents_all)
        label_all = torch.cat(label_all)

        return latents_all, label_all

    T_latents_A_train = gather_latents(s.MT, train_A_loader)
    S_latents_A_train = gather_latents(s, train_A_loader)

    T_latents_A_test = gather_latents(s.MT, test_A_loader)
    S_latents_A_test = gather_latents(s, test_A_loader)

    translate_model = Small_translate(single_dim=16, dropout=0.).cuda()
    translate_optimizer = torch.optim.Adam(translate_model.parameters(), lr=1e-3)

    translate_MLP = nn.Linear(neuron_amount['{}{}'.format(FLAGS.animal_trans,
                                                          FLAGS.day_trans)], 8).cuda()
    MLP_optimizer = torch.optim.Adam(translate_MLP.parameters(), lr=1e-3)

    helper = translate_helper_H(translate_model, translate_optimizer, translate_MLP, MLP_optimizer)
    helper.train_translate(T_latents_A_train, S_latents_A_train, T_latents_A_test, S_latents_A_test, total_epoch=200)

    T_latents_B_train = gather_latents(s.MT, train_B_loader)
    T_latents_B_test = gather_latents(s.MT, test_B_loader)

    helper.translate_MLP(T_latents_B_train, T_latents_B_test, total_epoch=200)
    # helper.test_translate_MLP(T_latents_B_test)

def distance(args, train_A_loader, test_A_loader, train_B_loader, test_B_loader):
    # distribution A from train and test
    # distribution B from train and test
    # logic:
    # load model A trained on A, get latents from data A and data B

    # compute distance within neurons in A as well as neurons in B

    from test_distance_funcs import MMD_loss
    from tqdm import tqdm
    from einops import rearrange

    data, label = next(iter(train_A_loader))
    print(data.shape) # why 6 is actually longer than 2 ?

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=data.shape[-1],
        neuron_dropout=FLAGS.neuron_dropout,
    ).cuda()

    # print(v)

    '''
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
    '''

    LOAD = 'ckpt_neural/VIT_S{}{}-0512-6-store/vit_T_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
    v.load_state_dict(torch.load(LOAD))

    def gather_latents(model, dataloader):
        latents_all = []
        label_all = []
        data_all = []

        model.eval()
        with torch.no_grad():
            for data, label in dataloader:
                data, label = data.cuda(), label.cuda()

                latents = model.translate_l(data)
                latents_all.append(latents)
                label_all.append(label)
                data_all.append(data)

        latents_all = torch.cat(latents_all)
        label_all = torch.cat(label_all)
        data_all = torch.cat(data_all)

        print(latents_all.shape, label_all.shape, data_all.shape)
        return (latents_all, label_all), data_all

    T_latents_A_train, data_A_train = gather_latents(v, train_A_loader)
    T_latents_B_train, data_B_train = gather_latents(v, train_B_loader)

    data_A_m = data_A_train
    label_A_m = T_latents_A_train[1]
    data_B_m = data_B_train
    label_B_m = T_latents_B_train[1]

    data_A_m_flat = rearrange(data_A_train, 'b t n -> (b t) n')
    label_A_m_flat = rearrange(T_latents_A_train[1], 'b t -> (b t)')
    data_B_m_flat = rearrange(data_B_train, 'b t n -> (b t) n')
    label_B_m_flat = rearrange(T_latents_B_train[1], 'b t -> (b t)')


    # print(T_latents_A_train[0].shape, T_latents_A_train[1].shape)
    # print(T_latents_B_train[0].shape, T_latents_B_train[1].shape)

    b1, n1, _ = T_latents_A_train[0].shape
    b2, n2, _ = T_latents_B_train[0].shape

    latentsA = T_latents_A_train[0].cpu().detach()
    latentsA_copy = latentsA.clone()
    latentsB = T_latents_B_train[0].cpu().detach()
    latentsB_copy = latentsB.clone()


    neuron_uniques_A = []
    for i in range(n1):
        _, unqiues = torch.unique(latentsA[:, i, :], return_counts=True, dim=0)
        if unqiues.shape[0] > 50:
            #print(i, unqiues)
            ...
            #pca = PCA(2)
            #A_latents = pca.fit_transform(latentsA[:, i, :])
            #plt.scatter(A_latents[:, 0], A_latents[:, 1])
            #plt.show()

        # print(i, unqiues.shape)
        neuron_uniques_A.append(unqiues.shape[0])

    neuron_uniques_B = []
    for i in range(n2):
        _, unqiues = torch.unique(latentsB[:, i, :], return_counts=True, dim=0)
        # print(i, unqiues.shape)
        neuron_uniques_B.append(unqiues.shape[0])

    maskA = torch.Tensor(neuron_uniques_A) > 980

    latentsA = latentsA[:, maskA, :]
    print(latentsA.shape)

    maskB = torch.Tensor(neuron_uniques_B) > 850
    latentsB = latentsB[:, maskB, :]
    print(latentsB.shape)

    def same_animal_map():
        score_map = torch.zeros(25, 25)
        from test_distance_funcs import MMD_loss
        from tqdm import tqdm

        MMD_helper = MMD_loss()

        for i in tqdm(range(latentsA.shape[1])):
            for j in range(latentsA.shape[1]):
                score = MMD_helper.compute(latentsA[:, i, :], latentsA[:, j, :])
                score_map[i, j] = score

        plt.imshow(score_map)
        plt.show()

    def diff_animal_map():
        score_map = torch.zeros(25, 25)
        MMD_helper = MMD_loss()

        for i in tqdm(range(latentsA.shape[1])):
            for j in range(latentsB.shape[1]):
                score = MMD_helper.compute(latentsA[:, i, :], latentsB[:b1, j, :])
                score_map[i, j] = score

        plt.imshow(score_map)
        plt.show()

    def plot(latentsA, id=17):
        pca = PCA(2)
        latentsA = pca.fit_transform(latentsA[:, id, :])

        plt.scatter(latentsA[:, 0], latentsA[:, 1],
                    c=rearrange(T_latents_A_train[1].cpu().detach(), 'b t -> (b t)'),
                    cmap='tab10')
        plt.show()

    # same_animal_map()
    # diff_animal_map()

    #plot(latentsA_copy, id=45)


    # save data for mehdi


    latents_A_m = latentsA_copy
    latents_B_m = latentsB_copy


    if not os.path.exists("cool_OT/{}{}to{}{}".format(FLAGS.animal, FLAGS.day, FLAGS.animal_trans, FLAGS.day_trans)):
        os.makedirs("cool_OT/{}{}to{}{}".format(FLAGS.animal, FLAGS.day, FLAGS.animal_trans, FLAGS.day_trans))

    np.savez('cool_OT/{}{}to{}{}/latents_M_0513_raw_data.npz'.format(FLAGS.animal, FLAGS.day, FLAGS.animal_trans, FLAGS.day_trans),
             data_A = data_A_m.cpu().detach(), data_B = data_B_m.cpu().detach(),
             label_A = label_A_m.cpu().detach(), label_B = label_B_m.cpu().detach(),
             data_A_flat=data_A_m_flat.cpu().detach(), data_B_flat=data_B_m_flat.cpu().detach(),
             label_A_flat=label_A_m_flat.cpu().detach(), label_B_flat=label_B_m_flat.cpu().detach(),
             latents_A = latents_A_m, latents_B = latents_B_m)



import scipy.signal as signal
import neo
import quantities as pq
from elephant.gpfa import GPFA

def transfer_benchmark(args, train_A_loader, test_A_loader, train_B_loader, test_B_loader, model='ndt'):

    if model == 'smoothing' or model == 'gpfa':
        data_A_train, label_A_train = [], []
        for data, label in train_A_loader:
            data_A_train.append(data)
            label_A_train.append(label)
        data_A_train, label_A_train = torch.cat(data_A_train, dim=0), torch.cat(label_A_train, dim=0)

        data_A_test, label_A_test = [], []
        for data, label in test_A_loader:
            data_A_test.append(data)
            label_A_test.append(label)
        data_A_test, label_A_test = torch.cat(data_A_test, dim=0), torch.cat(label_A_test, dim=0)

        data_B_train, label_B_train = [], []
        for data, label in train_B_loader:
            data_B_train.append(data)
            label_B_train.append(label)
        data_B_train, label_B_train = torch.cat(data_B_train, dim=0), torch.cat(label_B_train, dim=0)

        data_B_test, label_B_test = [], []
        for data, label in test_B_loader:
            data_B_test.append(data)
            label_B_test.append(label)
        data_B_test, label_B_test = torch.cat(data_B_test, dim=0), torch.cat(label_B_test, dim=0)

    if model == 'smoothing':

        # Smooth spikes
        kern_sd = 2
        window = signal.gaussian(kern_sd * 1, kern_sd, sym=True)
        window /= np.sum(window)
        filt = lambda x: np.convolve(x, window, 'same')

        data_A_train_latents = np.apply_along_axis(filt, 1, data_A_train)
        data_A_test_latents = np.apply_along_axis(filt, 1, data_A_test)
        data_B_train_latents = np.apply_along_axis(filt, 1, data_B_train)
        data_B_test_latents = np.apply_along_axis(filt, 1, data_B_test)

        print(data_A_train_latents.shape)

        from einops import rearrange
        from torch.utils.data import DataLoader, Dataset

        '''
        MLP_base = Transfer_MLP(neuron=data_A_train_latents.shape[-1], latent=16, dropout=0.2, cls=8).cuda()
        MLP_base_optim = torch.optim.Adam(MLP_base.parameters(), lr=1e-4)

        MLP_trans = Transfer_MLP_new_end(neuron_new=data_B_train_latents.shape[-1], dropout=0.2, cls=8).cuda()
        MLP_trans_optim = torch.optim.Adam(MLP_trans.parameters(), lr=1e-4)

        transfer_bchmk(MLP_base,
                       MLP_base_optim,
                       MLP_trans,
                       MLP_trans_optim,
                       data_A_train_latents,
                       data_A_test_latents,
                       data_B_train_latents,
                       data_B_test_latents,
                       label_A_train,
                       label_A_test,
                       label_B_train,
                       label_B_test)
        # logic:
        # train MLP_base on data_A_train_latents, test on data_A_test_latents
        # use latents produced by MLP_base, train a new MLP on data_B_train_latents, test on data_B_test_latents
        '''

        MLP = nn.Sequential(nn.Linear(data_B_train_latents.shape[-1], 8)).cuda()
        MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-4)

        total_epochs = 400
        progress_bar = tqdm(range(total_epochs), position=0, leave=True)
        crit = torch.nn.CrossEntropyLoss()

        class Simple_Trans(Dataset):
            def __init__(self, data, label):
                # [reps, labels]
                self.reps = torch.from_numpy(data).float()
                self.labels = label
                # print(self.reps.shape, self.labels.shape) # torch.Size([60000, 64]) torch.Size([60000])

            def __len__(self):
                return self.labels.shape[0]

            def __getitem__(self, idx):
                return self.reps[idx], self.labels[idx]

        # print(data_B_train_latents.shape, label_B_train.shape)

        train_loader = DataLoader(Simple_Trans(data=rearrange(data_B_train_latents, 'b t n -> (b t) n'),
                                               label=rearrange(label_B_train, 'b t -> (b t)')), batch_size=64)
        test_loader = DataLoader(Simple_Trans(data=rearrange(data_B_test_latents, 'b t n -> (b t) n'),
                                              label=rearrange(label_B_test, 'b t -> (b t)')), batch_size=64)

        def eval_MLP(MLP, loader):
            MLP.eval()
            running_loss = 0.
            right = []
            total = []
            with torch.no_grad():
                for x, label in loader:
                    x, label = x.cuda(), label.cuda()
                    preds = MLP(x)
                    loss = crit(preds, label)
                    running_loss += loss

                    _, pred_class = torch.max(preds, 1)
                    right.append((pred_class == label).sum().item())
                    total.append(label.size(0))

            MLP.train()
            return running_loss, sum(right) / sum(total)

        for epoch in progress_bar:
            right, total = [], []

            for x, label in train_loader:
                MLP.train()
                MLP_optim.zero_grad()
                x, label = x.cuda(), label.cuda()

                preds = MLP(x.float())
                loss = crit(preds, label)

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))

                loss.backward()
                MLP_optim.step()

            running_eval_loss, eval_clf = eval_MLP(MLP, test_loader)

            progress_bar.set_description(
                'Loss/train_clf: {}, Loss/eval_clf: {}'.format(sum(right) / sum(total), eval_clf))

    if model == 'gpfa':
        def array_to_spiketrains(array):
            """Converts trial x time x channel spiking arrays to list of list of neo.SpikeTrain"""
            array = array.numpy()
            stList = []
            # Loop through trials
            for trial in range(len(array)):
                trialList = []
                # Loop through channels
                for channel in range(array.shape[2]):
                    # Get spike times and counts
                    times = np.where(array[trial, :, channel])[0]
                    counts = array[trial, times, channel].astype(int)
                    train = np.repeat(times, counts)
                    # Create neo.SpikeTrain
                    st = neo.SpikeTrain(times * 100 * pq.ms, t_stop=array.shape[1] * 100 * pq.ms)
                    trialList.append(st)
                stList.append(trialList)
            return stList

        # Run conversion
        data_A_train = array_to_spiketrains(data_A_train)
        data_A_test = array_to_spiketrains(data_A_test)
        data_B_train = array_to_spiketrains(data_B_train)
        data_B_test = array_to_spiketrains(data_B_test)

        # Set parameters
        bin_size = 100 * pq.ms
        latent_dim = 2

        # Train GPFA on train data and apply on test data
        gpfa = GPFA(bin_size=bin_size, x_dim=latent_dim)
        train_A_factors = gpfa.fit_transform(data_A_train)
        eval_A_factors = gpfa.transform(data_A_test)

        print(train_A_factors.shape) # (1032,)

    if model == 'ndt':
        data_A, _ = next(iter(train_A_loader))
        data_B, _ = next(iter(train_B_loader))

        MLP_base = Transfer_NDT(neuron=data_A.shape[-1], dropout=0.2, num_classes=8).cuda()
        MLP_base_optim = torch.optim.Adam(MLP_base.parameters(), lr=1e-4)

        MLP_trans = Transfer_NDT_new_end(neuron_new=data_B.shape[-1], dropout=0.2, latents=data_A.shape[-1]*2).cuda()
        MLP_trans_optim = torch.optim.Adam(MLP_trans.parameters(), lr=1e-4)

        transfer_bdt_bchmk(MLP_base,
                       MLP_base_optim,
                       MLP_trans,
                       MLP_trans_optim,
                       train_A_loader,
                       test_A_loader,
                       train_B_loader,
                       test_B_loader)




def Visual_neuron(args, train_loader, test_loader):

    # old_path
    # LOAD = 'ckpt_neural/VIT_Smihi1-v1-store/vit_T_epoch399.pt'

    # ckpt_neural/VIT_Smihi1-test-norm/vit_T_epoch199.pt
    # LOAD = 'ckpt_neural/VIT_S{}{}-v1-store/vit_T_epoch199.pt'.format(FLAGS.animal, FLAGS.day)
    # LOAD = 'ckpt_neural/VIT_S{}{}-v1-store-limited/vit_T_epoch199.pt'.format(FLAGS.animal, FLAGS.day)

    # ViT-S path used
    # LOAD = 'ckpt_neural/VIT_S{}{}-160-2-store-lr/vit_T_epoch599.pt'.format(FLAGS.animal, FLAGS.day)
    # ViT-T path used
    # LOAD = 'ckpt_neural/VIT_T{}{}-160-2-store/vit_epoch199.pt'.format(FLAGS.animal, FLAGS.day)

    #LOAD = 'ckpt_neural/VIT_T{}{}-160-6-store/vit_epoch399.pt'.format(FLAGS.animal, FLAGS.day)
    # v.load_state_dict(torch.load(LOAD))
    # v.eval()
    LOAD = 'ckpt_neural/VIT_Smihi1-160-6-store/vit_T_epoch199.pt'

    data, label = next(iter(train_loader))

    v = Neural_ViT_T(
        num_classes=8,
        single_dim=FLAGS.neuron_dim,
        depth=FLAGS.T_depth,
        heads=FLAGS.T_heads,
        neuron=neuron_amount['mihi1'],
    ).cuda()

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

    v.load_state_dict(torch.load(LOAD))
    v.eval()

    latents_train = []
    labels_train = []
    raw_data = []

    for i, (x, label) in enumerate(train_loader):
        """data_with_temp shape like [b t n], label_with_temp shape like [b t]"""
        with torch.no_grad():
            raw_data.append(x)
            latents, small_latents = v.get_latent_t(x.cuda())  # b n t d
            latents = latents.detach().cpu()  # b n t d

            latents_train.append(latents)
            labels_train.append(label)

    raw_data = torch.cat(raw_data)
    print('raw', raw_data.shape) # torch.Size([167, 8, 163]) # [b t n]
    # neuron 14
    #for n_id in range(70):
    raw_data_i = raw_data[:, :, 14]  # [167, 8, 163]
    for time in range(6):
        print(time, len(torch.unique(raw_data_i[:, time])))

    latents_train = torch.cat(latents_train, dim=0)
    labels_train = torch.cat(labels_train, dim=0)

    latents_train = latents_train[::3]
    labels_train = labels_train[::3]

    neuron = 45

    pca = PCA(2)
    small_latents_train = pca.fit_transform(rearrange(latents_train, 'b n t d -> (b n t) d'))
    labels_train = rearrange(labels_train, 'b t -> (b t)')


    labels_neuron = torch.arange(0, neuron_amount['{}{}'.format(FLAGS.animal, FLAGS.day)])
    labels_neuron = repeat(labels_neuron, 'n -> b n t', b=latents_train.shape[0], t=6)
    labels_neuron = rearrange(labels_neuron, 'b n t -> (b n t)')

    small_latents_train = rearrange(small_latents_train, '(b n t) d -> b n t d', b=latents_train.shape[0], t=6)
    small_latents_train_neuron = small_latents_train[:, neuron, :, :]  # b t d
    for time in range(6):
        print(time, len(torch.unique(torch.Tensor(small_latents_train_neuron[:, time, 0]))))

    fig_new, ax_new = plt.subplots(1, 1)
    small_latents_train_neuron = rearrange(small_latents_train_neuron, 'b t d -> (b t) d')  # (b t) d

    print(small_latents_train_neuron.shape)
    scatter = ax_new.scatter(small_latents_train_neuron[:, 0], small_latents_train_neuron[:, 1], c=labels_train, cmap='tab10')
    legend = ax_new.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
    ax_new.add_artist(legend)

    plt.show()

    def draw_mihi1_on_mihi1():
        pca = PCA(2)
        fig_new, ax_new = plt.subplots(1, 1, figsize=(6, 6))
        latents_train_neuron = latents_train[:, neuron, :, :]
        small_latents_train_neuron = pca.fit_transform(rearrange(latents_train_neuron, 'b t d -> (b t) d'))

        scatter = ax_new.scatter(small_latents_train_neuron[:, 0], small_latents_train_neuron[:, 1], c=labels_train,
                                 cmap='tab10')
        legend = ax_new.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
        ax_new.add_artist(legend)

        center_connect_x = []
        center_connect_y = []
        for label_i in range(8):
            mask_i = labels_train == label_i
            list_points = small_latents_train_neuron[mask_i]
            center_connect_x.append(np.mean(list_points, axis=0)[0])
            center_connect_y.append(np.mean(list_points, axis=0)[1])

        ax_new.scatter(center_connect_x, center_connect_y, c='k', marker='x', s=60)

        ax_new.set_title("neuron is {}".format(neuron))

        plt.savefig('neuron_same_model_cmap.eps')

    def draw_mihi1_on_chewie1():
        neuron = 49
        pca = PCA(2)
        fig_new, ax_new = plt.subplots(1, 1, figsize=(6, 6))
        latents_train_neuron = latents_train[:, neuron, :, :]
        small_latents_train_neuron = pca.fit_transform(rearrange(latents_train_neuron, 'b t d -> (b t) d'))

        scatter = ax_new.scatter(small_latents_train_neuron[:, 0], small_latents_train_neuron[:, 1], c=labels_train,
                                 cmap='tab10')
        legend = ax_new.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
        ax_new.add_artist(legend)

        center_connect_x = []
        center_connect_y = []
        for label_i in range(8):
            mask_i = labels_train == label_i
            list_points = small_latents_train_neuron[mask_i]
            center_connect_x.append(np.mean(list_points, axis=0)[0])
            center_connect_y.append(np.mean(list_points, axis=0)[1])

        ax_new.scatter(center_connect_x, center_connect_y, c='k', marker='x', s=60)

        ax_new.set_title("neuron is {}".format(neuron))

        #plt.show()

        plt.savefig('neuron_diff_model_cmap.eps')

    draw_mihi1_on_mihi1()



    def check_neuron_pattern(small_latents_train):
        fig_new, ax_new = plt.subplots(10, 8, figsize=(10, 15), sharex=True, sharey=True)
        small_latents_train = rearrange(small_latents_train, '(b n t) d -> b n t d', b=latents_train.shape[0], t=8)

        for neuron_i, neuron_id in enumerate(range(70, 80)):
            #neuron_id = 7
            small_latents_train_neuron = small_latents_train[:, neuron_id, :, :] # b t d
            small_latents_train_neuron = rearrange(small_latents_train_neuron, 'b t d -> (b t) d') # (b t) d
            for i in range(8):
                labels_mask = labels_train == i
                small_latents_train_neuron_i = small_latents_train_neuron[labels_mask]
                ax_new[neuron_i, i].scatter(small_latents_train_neuron_i[:, 0], small_latents_train_neuron_i[:, 1], c='k', marker='x')
        plt.show()

    PLOT = False
    if PLOT:
        # small_latents_train_plt = rearrange(small_latents_train, 'b n t d -> (b n t) d')

        #random_selection = torch.randperm(small_latents_train_plt.shape[0])
        #train_set = random_selection[:20000]
        #small_latents_train_plt = small_latents_train_plt[train_set]

        clustering = KMeans(n_clusters=9, random_state=0).fit(rearrange(latents_train, 'b n t d -> (b n t) d'))
        cluster_labels = clustering.labels_

        import pickle
        # Its important to use binary mode
        #knnPickle = open('knnmihi1', 'wb')
        # source, destination
        #pickle.dump(clustering, knnPickle)
        # load the model from disk
        #clustering = pickle.load(open('knnmihi1', 'rb'))
        #cluster_labels = clustering.predict(small_latents_train_plt)

        print(labels_train.shape, small_latents_train.shape) # torch.Size([217768]) (217768, 2)

        # small_latents_train = small_latents_train[labels_mask]

        fig, axes = plt.subplots(1, 1)
        scatter = axes.scatter(small_latents_train[:, 0], small_latents_train[:, 1], c=cluster_labels)
        legend = axes.legend(*scatter.legend_elements(), loc="lower right", title="Classes")

        axes.add_artist(legend)
        plt.show()

        # small_latents_train = rearrange(small_latents_train, '(b n t) d -> b n t d', b=latents_train.shape[0], t=8)

        #for neuron_id in range(20):
        #    # print(neuron_id)
        #    fig_new, ax_new = plt.subplots(1, 1)
        #    small_latents_train_neuron = small_latents_train[:, neuron_id, :, :]
        #    small_latents_train_neuron = rearrange(small_latents_train_neuron, 'b t d -> (b t) d')
            # print(small_latents_train_neuron, )
        #    ax_new.scatter(small_latents_train_neuron[:, 0], small_latents_train_neuron[:, 1], c='k', marker='x')
        #    plt.show()

    '''
    fig2, ax = plt.subplots(ncols=1, nrows=8, figsize=(120/4, 9*2))
    spectrogram_recorder = []
    for direction_i in range(8):

        label_mask = labels_train != direction_i # tensor of shape (b n t)

        spectrogram = np.ones((9, 160))  # neurons x clusters
        for cluster_i in range(9):
            small_latents_cls = small_latents_train.clone()
            mask = cluster_labels == cluster_i
            small_latents_cls[mask] = 1e5

            # force label masking
            small_latents_cls[label_mask] = 0

            small_latents_cls = rearrange(small_latents_cls, '(b n t) d -> b n t d', n=160, t=8)
            posi = (small_latents_cls == 1e5).nonzero(as_tuple=True)

            unique_neurons = torch.unique(posi[1])
            print(unique_neurons.shape, cluster_i)
            spectrogram[cluster_i, unique_neurons] = 0

        ax[direction_i].imshow(spectrogram)
        spectrogram_recorder.append(spectrogram)

    plt.show()

    # direction 0 signiture neurons, ranking
    spectrogram_diff0 = np.zeros((9, 160))
    for direction_i in range(1, 8):
        spectrogram_diff = np.zeros((9, 160))
        spectrogram_diff[spectrogram_recorder[0] != spectrogram_recorder[1]] = 1

        spectrogram_diff0 = spectrogram_diff0 + spectrogram_diff

    fig3, ax = plt.subplots(ncols=1, nrows=1, figsize=(120, 9))
    ax.imshow(spectrogram_diff0)
    plt.show()


    mask = (small_latents_train[:, :, :, 0] < 0) & (small_latents_train[:, :, :, 0] > -9)
    small_latents_train[mask] = 1e5
    small_latents_train = torch.Tensor(small_latents_train)

    posi = (small_latents_train == 1e5).nonzero(as_tuple=True)

    print(torch.unique(posi[1]))
    print((torch.unique(posi[1])).shape)
    '''


if __name__ == "__main__":
    print(f'PyTorch version: {torch.__version__}')

    app.run(run)
    # 80 turns out to be better than 120 and 40
