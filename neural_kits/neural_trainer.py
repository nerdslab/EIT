import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from my_transformers.models import *
from my_transformers.tasks import linear_clf
from neural_kits.neural_tasks import angle_linear_clf, transfer_mlp, transfer_mlp_ssl, gen_linear_clf

class vit_neural_learner(pl.LightningModule):
    """this is the normal ViT learner"""
    def __init__(self,
                 vit,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 check_clf=5,
                 ):
        super().__init__()
        self.net = vit
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label
        self.check_clf = check_clf

        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, label = batch

        #if x.shape[1] == 2:
        #    if torch.rand(1) < 0.5:
        #        x[:, 1, :] = x[:, 0, :] + torch.rand(x[:, 0, :].shape, device=x.device) - 0.5

        preds, _ = self.forward(x)
        if self.reshape_label:
            label = rearrange(label, 'b t -> (b t)')

        if type(preds) == type({'S': x}):
            alpha = 0.5
            # print(preds['S'].shape, preds['T'].shape)
            loss = self.crit(preds['S'], label) + alpha * self.crit(preds['T'], label)
        else:
            loss = self.crit(preds, label)

        self.logger.log_metrics({'Loss/train_loss': loss,
                                 }, step=self.global_step)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % self.check_clf == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                right, total = [], []
                for x, label in self.add['test_loader']:

                    #if x.shape[1] == 2:
                    #    x[:, 1, :] = x[:, 0, :]

                    x, label = x.cuda(), label.cuda()
                    preds, _ = self.net(x)
                    if self.reshape_label:
                        label = rearrange(label, 'b t -> (b t)')
                    if type(preds) == type({'S': x}):
                        loss = self.crit(preds['S'], label) + self.crit(preds['T'], label)
                    else:
                        loss = self.crit(preds, label)

                    running_eval_loss += loss

                    if type(preds) == type({'S': x}):
                        _, pred_class = torch.max(preds['S'], 1)
                    else:
                        _, pred_class = torch.max(preds, 1)

                    right.append((pred_class == label).sum().item())
                    total.append(label.size(0))
            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)
            self.logger.log_metrics({'Acc/eval_clf': sum(right) / sum(total)}, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            if 'save_dict' in self.add.keys():
                torch.save(self.net.state_dict(), '{}/{}/vit_epoch{}.pt'.format(self.add['save_dict'], self.TB_LOG_NAME, self.current_epoch))

            else:
                torch.save(self.net.state_dict(), 'ckpt_neural/{}/vit_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))
                if 'save_t' in self.add.keys():
                    if self.add['save_t']:
                        torch.save(self.net.MT.state_dict(),
                                   'ckpt_neural/{}/vit_T_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.net.train()

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)

class ndt_ssl_neural_learner(pl.LightningModule):
    """this is the normal ViT learner"""
    def __init__(self,
                 vit,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 check_clf=5,
                 ):
        super().__init__()
        self.net = vit
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label
        self.check_clf = check_clf

    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, label = batch

        if x.shape[1] == 2:
            if torch.rand(1) < 0.5:
                x[:, 1, :] = x[:, 0, :] + torch.rand(x[:, 0, :].shape, device=x.device) - 0.5

        preds, _ = self.forward(x)
        loss = preds

        self.logger.log_metrics({'Loss/train_loss': loss,
                                 }, step=self.global_step)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % self.check_clf == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                for x, label in self.add['test_loader']:
                    x, label = x.cuda(), label.cuda()
                    preds, _ = self.net(x)
                    loss = preds
                    running_eval_loss += loss

            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)

        if (self.current_epoch + 1) % 50 == 0:
            MLP = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(self.add['neuron_shape'], 8)).cuda()
            MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-3)
            TRANS = transfer_mlp(self.net, MLP, MLP_optim,
                                 self.add['train_loader'], self.add['test_loader'], self.logger, total_epoch=200)
            self.logger.log_metrics({'Acc/final_loss': TRANS.final_loss,
                                     'Acc/final_perf': TRANS.final_perf}, step=self.current_epoch)
            # TRANS.final_loss, TRANS.final_perf

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            if 'save_dict' in self.add.keys():
                torch.save(self.net.state_dict(), '{}/{}/vit_epoch{}.pt'.format(self.add['save_dict'], self.TB_LOG_NAME, self.current_epoch))

            else:
                torch.save(self.net.state_dict(), 'ckpt_neural/{}/vit_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))
                if 'save_t' in self.add.keys():
                    if self.add['save_t']:
                        torch.save(self.net.MT.state_dict(),
                                   'ckpt_neural/{}/vit_T_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.net.train()

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)


class vit_neural_learner_G(pl.LightningModule):
    """this is the normal ViT learner"""
    def __init__(self,
                 vit,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 check_clf=5,
                 ):
        super().__init__()
        self.net = vit
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label
        self.check_clf = check_clf

        # self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, label = batch
        loss, _ = self.forward(x)
        self.logger.log_metrics({'Loss/train_loss': loss,
                                 }, step=self.global_step)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % self.check_clf == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                for x, label in self.add['test_loader']:
                    x, label = x.cuda(), label.cuda()
                    loss, _ = self.net(x)
                    running_eval_loss += loss
            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)
            #self.logger.log_metrics({'Acc/eval_clf': sum(right) / sum(total)}, step=self.current_epoch)

        if (self.current_epoch + 1) % 2 == 0:
            MLP = nn.Sequential(nn.Linear(self.add['neuron_dim'], 8)).cuda()
            MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-3)
            TRANS = gen_linear_clf(MLP, MLP_optim, self.net,
                                   self.add['train_loader'], self.add['test_loader'], writer=self.logger, num_epochs=400)
            self.logger.log_metrics({'Acc/eval_clf': TRANS.best_number}, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            pass

        self.net.train()

class vit_neural_SSL_learner(pl.LightningModule):
    """this is the SSL ViT learner"""
    def __init__(self,
                 v,
                 s,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 check_clf=5,
                 ):
        super().__init__()
        self.fixed_v = v
        self.s = s

        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label
        self.check_clf = check_clf

    def forward(self, trans_x, small_trans_x):
        return self.s(trans_x, small_trans_x, ssl=True)

    def training_step(self, batch, batch_idx):
        x, label = batch

        self.fixed_v.eval()
        with torch.no_grad():
            trans_x, small_trans_x = self.fixed_v.get_latent_t(x)

        recon_loss = self.forward(trans_x, small_trans_x)
        # print(recon_loss)
        self.logger.log_metrics({'Loss/train_loss': recon_loss,
                                 }, step=self.global_step)
        return {'loss': recon_loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.s.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % 1 == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                for x, _ in self.add['test_loader']:
                    # print(x.shape, label.shape)
                    x = x.cuda()
                    trans_x, small_trans_x = self.fixed_v.get_latent_t(x)
                    loss = self.s(trans_x, small_trans_x, ssl=True)
                    running_eval_loss += loss
            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)

        if (self.current_epoch + 1) % 100 == 0:
            MLP = nn.Sequential(nn.Dropout(0.2),
                nn.Linear(self.add['neuron_shape'], 8)).cuda()
            MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-3)
            TRANS = transfer_mlp_ssl(self.fixed_v, self.s,
                             MLP, MLP_optim,
                             self.add['train_loader'], self.add['test_loader'], self.logger, total_epoch=400)
            self.logger.log_metrics({'Acc/final_loss': TRANS.final_loss,
                                     'Acc/final_perf': TRANS.final_perf}, step=self.current_epoch)
            # TRANS.final_loss, TRANS.final_perf

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            torch.save(self.s.state_dict(), 'ckpt_neural/{}/vit_ssl_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.s.train()

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)

class vit_neural_learner_trans(pl.LightningModule):
    """this is the ViT trans learner -- capable of transfer learning based on multiple animals"""
    def __init__(self,
                 vit,
                 datasets_info,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 ):
        super().__init__()
        self.net = vit
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label

        self.data_amount = len(datasets_info['animals'])
        self.data_info = datasets_info
        self.data_name = ['{}{}'.format(datasets_info['animals'][i],
                                        datasets_info['days'][i]) for i in range(self.data_amount)]

        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, img, bottom):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img, head=bottom)

    def training_step(self, batch, batch_idx):
        # x, label = batch

        loss_all = {'chewie1': 0., 'chewie2': 0., 'mihi1': 0., 'mihi2': 0.}
        loss = 0.

        for i in range(self.data_amount):
            x, label = batch[i]
            preds, _ = self.forward(x, self.data_name[i])
            if self.reshape_label:
                label = rearrange(label, 'b t -> (b t)')
            loss_i = self.crit(preds, label)

            loss_all[self.data_name[i]] = loss_all[self.data_name[i]] + loss_i
            loss = loss + loss_i

        self.logger.log_metrics({'Loss/train_loss': loss}, step=self.global_step)

        self.logger.log_metrics({'Transfer/train_chewie1': loss_all['chewie1'],
                                 'Transfer/train_chewie2': loss_all['chewie2'],
                                 'Transfer/train_mihi1': loss_all['mihi1'],
                                 'Transfer/train_mihi2': loss_all['mihi2'],
                                 }, step=self.global_step)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # do eval loss evaluation
        if (self.current_epoch + 1) % 1 == 0:

            eval_loss_all = {'chewie1': 0., 'chewie2': 0., 'mihi1': 0., 'mihi2': 0.}
            eval_correct_all = {'chewie1': [], 'chewie2': [], 'mihi1': [], 'mihi2': []}
            eval_data_all = {'chewie1': [], 'chewie2': [], 'mihi1': [], 'mihi2': []}
            eval_loss, correct_all, data_all = 0., 0., 0.

            with torch.no_grad():
                # test_loader_i = self.add['test_loader'][i]
                for batch in self.add['test_loader']:
                    for i, dataname in enumerate(self.data_name):
                        x, label = batch[i]
                        x, label = x.cuda(), label.cuda()

                        preds, _ = self.net(x, dataname)
                        if self.reshape_label:
                            label = rearrange(label, 'b t -> (b t)')
                        loss = self.crit(preds, label)
                        eval_loss_all[dataname] += loss

                        _, pred_class = torch.max(preds, 1)
                        eval_correct_all[dataname].append((pred_class == label).sum().item())
                        eval_data_all[dataname].append(label.size(0))

            eval_loss = eval_loss_all['chewie1'] + eval_loss_all['chewie2'] + eval_loss_all['mihi1'] + eval_loss_all['mihi2']
            for i, dataname in enumerate(self.data_name):
                right = eval_correct_all[dataname]
                total = eval_data_all[dataname]
                correct_all = correct_all + sum(right)
                data_all = data_all + sum(total)

            self.logger.log_metrics({'Loss/eval_loss': eval_loss}, step=self.current_epoch)
            self.logger.log_metrics({'Acc/eval_clf': correct_all/data_all}, step=self.current_epoch)

            self.logger.log_metrics({'Transfer/eval_chewie1': eval_loss_all['chewie1'],
                                     'Transfer/eval_chewie2': eval_loss_all['chewie2'],
                                     'Transfer/eval_mihi1': eval_loss_all['mihi1'],
                                     'Transfer/eval_mihi2': eval_loss_all['mihi2'],
                                     }, step=self.current_epoch)

            # print(eval_correct_all, eval_data_all)

            self.logger.log_metrics({'Transfer_clf/eval_chewie1': sum(eval_correct_all['chewie1'])/sum(eval_data_all['chewie1']),
                                     'Transfer_clf/eval_chewie2': sum(eval_correct_all['chewie2'])/sum(eval_data_all['chewie2']),
                                     'Transfer_clf/eval_mihi1': sum(eval_correct_all['mihi1'])/sum(eval_data_all['mihi1']),
                                     'Transfer_clf/eval_mihi2': sum(eval_correct_all['mihi2'])/sum(eval_data_all['mihi2']),
                                     }, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            if 'save_dict' in self.add.keys():
                torch.save(self.net.state_dict(), '{}/{}/vit_epoch{}.pt'.format(self.add['save_dict'], self.TB_LOG_NAME, self.current_epoch))

            else:
                torch.save(self.net.state_dict(), 'ckpt_neural/{}/vit_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.net.train()

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)

class vit_neural_learner_dirc(pl.LightningModule):
    """this is the transfer direction ViT learner, support s and t learning"""
    def __init__(self,
                 vit,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 ):
        super().__init__()
        self.net = vit
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label

        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, label = batch

        preds, _ = self.forward(x)
        if self.reshape_label:
            label = rearrange(label, 'b t -> (b t)')
        # print(preds.shape, label.shape)

        if type(preds) == type({'S': x}):
            alpha = 0.5
            # print(preds['S'].shape, preds['T'].shape)
            loss = self.crit(preds['S'], label) + alpha * self.crit(preds['T'], label)
        else:
            loss = self.crit(preds, label)
        # loss = self.crit(preds, label)

        self.logger.log_metrics({'Loss/train_loss': loss,
                                 }, step=self.global_step)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        '''
        if (self.current_epoch + 1) % 5 == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                right, total = [], []
                for x, label in self.add['test_loader']:
                    # print(x.shape, label.shape)
                    # torch.Size([16, 2, 163]) torch.Size([16, 2])

                    x, label = x.cuda(), label.cuda()
                    preds, _ = self.net(x)
                    if self.reshape_label:
                        label = rearrange(label, 'b t -> (b t)')

                    if type(preds) == type({'S': x}):
                        alpha = 0.5
                        # print(preds['S'].shape, preds['T'].shape)
                        loss = self.crit(preds['S'], label) + alpha * self.crit(preds['T'], label)
                    else:
                        loss = self.crit(preds, label)
                    # loss = self.crit(preds, label)
                    running_eval_loss += loss

                    _, pred_class = torch.max(preds, 1)
                    right.append((pred_class == label).sum().item())
                    total.append(label.size(0))
            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)
            self.logger.log_metrics({'Acc/eval_clf': sum(right) / sum(total)}, step=self.current_epoch)
            
        '''


        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % 20 == 0:

            data, label = next(iter(self.add['test_loader_trainsplit']))

            MLP = nn.Sequential(nn.Dropout(self.add['neuron_dropout']),
                                nn.Linear(data.shape[-1], self.add['test_cls'])).cuda()
            # GRU special MLP
            #MLP = nn.Sequential(nn.Dropout(self.add['neuron_dropout']),
            #                   nn.Linear(160, self.add['test_cls'])).cuda()

            MLP_optim = torch.optim.Adam(MLP.parameters(), lr=1e-4)
            MLP_trainer = transfer_mlp(self.net, MLP, MLP_optim,
                                       self.add['test_loader_trainsplit'],
                                       self.add['test_loader_testsplit'],
                                       self.logger, total_epoch=400)

            self.logger.log_metrics({'Loss/eval_loss': MLP_trainer.final_loss}, step=self.current_epoch)
            self.logger.log_metrics({'Acc/eval_clf': MLP_trainer.final_perf}, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            ...

        self.net.train()

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)


class vit_neural_learner_multi(pl.LightningModule):
    """need editing"""
    def __init__(self,
                 vit,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None,
                 reshape_label=False,
                 ):
        super().__init__()
        self.net = vit
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add
        self.reshape_label = reshape_label

        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, label = batch

        preds, _ = self.forward(x)
        if self.reshape_label:
            label = rearrange(label, 'b t -> (b t)')
        # print(preds.shape, label.shape)
        loss = self.crit(preds, label)

        self.logger.log_metrics({'Loss/train_loss': loss,
                                 }, step=self.global_step)
        return {'loss': loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % 1 == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                right, total = [], []
                for x, label in self.add['test_loader']:
                    # print(x.shape, label.shape)
                    x, label = x.cuda(), label.cuda()
                    preds, _ = self.net(x)
                    if self.reshape_label:
                        label = rearrange(label, 'b t -> (b t)')
                    loss = self.crit(preds, label)
                    running_eval_loss += loss

                    _, pred_class = torch.max(preds, 1)
                    right.append((pred_class == label).sum().item())
                    total.append(label.size(0))
            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)
            self.logger.log_metrics({'Acc/eval_clf': sum(right) / sum(total)}, step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            if 'save_dict' in self.add.keys():
                torch.save(self.net.state_dict(), '{}/{}/vit_epoch{}.pt'.format(self.add['save_dict'], self.TB_LOG_NAME, self.current_epoch))

            else:
                torch.save(self.net.state_dict(), 'ckpt_neural/{}/vit_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.net.train()

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)

class mae_neural_learner(pl.LightningModule):
    """this is the normal MAE learner,
    the class designed here is used for tiny-imagenet training
    the forward function would take one image only"""
    def __init__(self,
                 vit,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None):
        super().__init__()
        self.net = vit

        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add


    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, label = batch
        (pred_pixel_values, masked_patches), encoded_tokens, _ = self.forward(x)

        loss = self.loss_function(masked_patches, pred_pixel_values)
        self.logger.log_metrics({'LossMAE/train_loss': loss
                                 }, step=self.global_step)
        return {'loss': loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % 1 == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                for x, _ in self.add['test_loader']:
                    x = x.cuda()
                    (pred_pixel_values, masked_patches), encoded_tokens, unshuffle = self.net(x)

                    loss = self.loss_function(masked_patches, pred_pixel_values)

                    running_eval_loss += loss
            self.logger.log_metrics({'LossMAE/eval_loss': running_eval_loss}, step=self.current_epoch)
            #self.logger.log_metrics({'Acc/eval_clf': sum(right) / sum(total)}, step=self.current_epoch)

        self.net.masking_ratio = 0.0

        # do classification evaluation
        if (self.current_epoch + 1) % 5 == 0:
            # method 1: train a mlp based on repres (SSL)
            # method 2: re-train vit based on labels (Model pre-training)
            # apply method 1 first.
            clf = torch.nn.Sequential(torch.nn.Linear(self.add['en_dim'], 8)).to('cuda')
            clf_optimizer = torch.optim.Adam(clf.parameters(), lr=0.005, weight_decay=1e-5)

            clf_trainer = linear_clf(clf, clf_optimizer, self.net,
                                     self.add['train_loader'], self.add['test_loader'],
                                     num_epochs=200)

            self.logger.log_metrics({'SSL_clf/acc_train': clf_trainer.train_acc,
                                     'SSL_clf/acc_test': clf_trainer.test_acc}, step=self.current_epoch)

        if (self.current_epoch + 1) % 20 == 0:
            # rotation classification for SSL training
            # not good whatsoever
            clf = torch.nn.Sequential(torch.nn.Linear(self.add['en_dim'], 2)).to('cuda')
            clf_optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

            clf_trainer = angle_linear_clf(clf, clf_optimizer, self.net,
                                           self.add['train_loader'], self.add['test_loader'],
                                           num_epochs=200)
            self.logger.log_metrics({'SSL_clf/Angle_acc_train': clf_trainer.train_acc[0],
                                     'SSL_clf/Angle_acc_test': clf_trainer.test_acc[0]}, step=self.current_epoch)

        self.net.masking_ratio = self.add['masking_ratio']

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            torch.save(self.net.state_dict(),
                       'ckpt_neural/{}/mae_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.net.train()

    @staticmethod
    def l2(x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return (2 - 2 * (x * y).sum(dim=-1))

    @staticmethod
    def loss_function(x, x_recon, distribution='poisson'):
        batch_size = x.size(0)  # [256 B, 163]
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
        elif distribution == 'weighted_bernoulli':
            weight = torch.tensor([0.1, 0.9]).to("cuda")  # just a label here
            weight_ = torch.ones(x.shape).to("cuda")
            weight_[x <= 0.5] = weight[0]
            weight_[x > 0.5] = weight[1]
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none')
            recon_loss = torch.sum(weight_ * recon_loss).div(batch_size)
        elif distribution == 'gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        elif distribution == 'poisson':
            # print((x - x_recon * torch.log(x)).shape)
            x_recon.clamp(min=1e-7, max=1e7)
            recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
        elif distribution == 'poisson2':
            # layer = nn.Softplus()
            # x_recon = layer(x_recon)
            x_recon = x_recon + 1e-7
            recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
        elif distribution == 'mse':
            recon_loss = F.mse_loss(x_recon, x)
        else:
            raise NotImplementedError

        return recon_loss

    @staticmethod
    def compute_accuracy(pred, label):
        right, total = [], []
        right.append((pred == label).sum().item())
        total.append(label.size(0))
        return sum(right) / sum(total)
