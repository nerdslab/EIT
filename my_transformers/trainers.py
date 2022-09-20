import os
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from my_transformers.models import *
from my_transformers.tasks import linear_clf
from my_transformers.plots import plt_recon, plt_trans, plt_tiny

class MLP(nn.Module):
    def __init__(self, dim=192, num_classes=20):
        super().__init__()
        self.layer = nn.Linear(dim, num_classes)

    def forward(self, rep):
        return self.layer(rep)


class MAE_learner(pl.LightningModule):
    """this is the normal MAE learner,
    the class designed here is used for tiny-imagenet training
    the forward function would take one image only"""
    def __init__(self,
                 vit,
                 mae,
                 augmentor,
                 LR,
                 TB_LOG_NAME,
                 SAVE=200,
                 add=None):
        super().__init__()
        self.vit = vit
        self.net = mae
        self.augmentor = augmentor
        self.LR = LR
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.add = add

    def forward(self, img):
        if self.augmentor is not None:
            img = self.augmentor.augment(img)
        return self.net(img)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss, _, _ = self.forward(x)

        self.logger.log_metrics({'Loss/train_loss': loss,
                                 }, step=self.global_step)
        return {'loss': loss}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.vit.eval()
        self.net.eval()

        # do eval loss evaluation
        # don't do it within validation_step bc it will block clf later.
        if (self.current_epoch + 1) % 5 == 0:
            running_eval_loss = 0.0
            with torch.no_grad():
                for x, _ in self.add['test_loader']:
                    loss, _, _ = self.net(x)
                    running_eval_loss += loss
            self.logger.log_metrics({'Loss/eval_loss': running_eval_loss}, step=self.current_epoch)

        # do classification
        if (self.current_epoch + 1) % 1 == 0:

            clf_net = torch.nn.Sequential(torch.nn.Linear(self.add['mlp_dim'], 20)).to('cuda')
            clf_opt = torch.optim.Adam(clf_net.parameters(), lr=1e-3)
            CLF = linear_clf(clf_net, clf_opt, self.net, self.add['train_loader'], self.add['test_loader'])
            self.logger.log_metrics({'LinearClf/train': CLF.train_acc,
                                     'LinearClf/test': CLF.test_acc,
                                     'LinearClf/best_numb':  CLF.best_number,
                                     }, step=self.current_epoch)
        # recon visual
        img_train, img_test, recon_train, recon_test = self.get_plot_items(self.net,
                                                                           self.add['batch_train'],
                                                                           self.add['batch_test'])
        self.logger.experiment.add_figure('reconstruction train', plt_tiny(img_train, recon_train),
                                          global_step=self.current_epoch)
        self.logger.experiment.add_figure('reconstruction test', plt_tiny(img_test, recon_test),
                                          global_step=self.current_epoch)

        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            torch.save(self.vit.state_dict(), 'ckpt/{}/vit_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))
            torch.save(self.net.state_dict(), 'ckpt/{}/mae_epoch{}.pt'.format(self.TB_LOG_NAME, self.current_epoch))

        self.vit.train()
        self.net.train()

    @staticmethod
    def get_plot_items(mae, batch_train, batch_test):
        with torch.no_grad():
            img_train, _ = batch_train
            img_test, _ = batch_test

            _, _, recon_train = mae(img_train.cuda())
            _, _, recon_test = mae(img_test.cuda())

            return img_train, img_test, recon_train, recon_test


class transMAE_learner(pl.LightningModule):
    """this is the translation MAE learner,
    the class designed here is used for e.g. facades training
    the forward function would take two images"""
    def __init__(self,
                 mae,
                 augmentor,
                 TB_LOG_NAME,
                 SAVE,
                 LR,
                 add=None):
        super().__init__()
        self.net = mae
        self.augmentor = augmentor
        self.TB_LOG_NAME, self.SAVE = TB_LOG_NAME, SAVE
        self.LR = LR
        self.add = add

    def forward(self, imgA, imgB):
        if self.augmentor is not None:
            imgA, imgB = self.augmentor(imgA, imgB)
        return self.net(imgA, imgB)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self.forward(x["A"], x["B"])

        self.logger.log_metrics({'Loss/total': loss,
                                 }, step=self.global_step)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LR)

    def on_epoch_end(self):
        self.net.eval()
        # show reconstructed activities
        ...
        # save the network
        if (self.current_epoch + 1) % self.SAVE == 0:
            torch.save(self.net.state_dict(), os.path.join("ckpt", self.TB_LOG_NAME, "epoch{}.pth".format(self.current_epoch)))

        self.net.train()