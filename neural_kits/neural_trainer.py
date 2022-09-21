
import pytorch_lightning as pl

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from my_transformers.models import *
from neural_kits.neural_tasks import transfer_mlp

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
