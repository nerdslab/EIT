import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Simple_Trans(Dataset):
    def __init__(self, data, transform=None):
        # [reps, labels]
        self.reps = data[0]
        self.labels = data[1]
        # print(self.reps.shape, self.labels.shape) # torch.Size([60000, 64]) torch.Size([60000])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.reps[idx], self.labels[idx]

class angle_linear_clf(object):
    def __init__(self,
                 clf_net,
                 clf_opt,
                 mae,
                 train_loader,
                 test_loader,
                 device='cuda',
                 batch_size=1024,
                 num_epochs=50,
                 disable_tqdm=False,
                 writer=None,
                 ):

        self.clf = clf_net
        self.opt = clf_opt

        self.model = mae
        self.best_number = 0.0

        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.disable_tqdm = disable_tqdm
        self.writer = writer

        self.data_train = Simple_Trans(self.compute_rep(train_loader))
        self.data_train = DataLoader(self.data_train, batch_size=batch_size)
        self.data_test = Simple_Trans(self.compute_rep(test_loader))
        self.data_test = DataLoader(self.data_test, batch_size=batch_size)

        self.train_angle_layer()

        self.train_acc = self.compute_angle_acc(self.data_train)
        self.test_acc = self.compute_angle_acc(self.data_test)

    def compute_rep(self, dataloader):
        reps, labels = [], []
        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            labels.append(label)
            # labels.append(label)

            # forward
            with torch.no_grad():
                #_, representation, _ = self.model(x)
                #reps.append(representation["A"].detach().cpu())
                #reps.append(representation["B"].detach().cpu())
                _, representation, _ = self.model(x)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        # self.net.train()
        return [reps, labels]

    def compute_angle_acc(self, dataloader):
        self.clf.eval()

        acc = []
        delta_acc = []

        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_cos_sin = self.clf(x).detach().clone()
            # compute accuracy
            pred_angles = torch.atan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
            pred_angles[pred_angles < 0] = pred_angles[pred_angles < 0] + 2 * np.pi

            angles = (2 * np.pi / 8 * label)[:, np.newaxis]
            diff_angles = torch.abs(pred_angles - angles.squeeze())
            diff_angles[diff_angles > np.pi] = torch.abs(diff_angles[diff_angles > np.pi] - 2 * np.pi)

            acc = (diff_angles < (np.pi / 8)).sum()
            acc = acc.item() / x.size(0)
            delta_acc = (diff_angles < (3 * np.pi / 16)).sum()
            delta_acc = delta_acc.item() / x.size(0)

        self.clf.train()
        return acc, delta_acc

    def train_angle_layer(self):
        class_criterion = torch.nn.MSELoss()
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm, position=0, leave=True)
        for epoch in progress_bar:
            for x, label in self.data_train:
                self.clf.train()
                self.opt.zero_grad()

                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.clf(x)
                angles = (2 * np.pi / 8 * label)[:, np.newaxis]
                cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)

                #print(pred_class, cos_sin)
                loss = class_criterion(pred_class, cos_sin)

                # backward
                loss.backward()
                self.opt.step()

            curr_number_ref, _ = self.compute_angle_acc(self.data_train)
            curr_number, _ = self.compute_angle_acc(self.data_test)
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer is not None:
                #self.writer.log_metrics({'CLFtraining/val': curr_number}, step=epoch)
                self.writer.log_metrics({'CLFtraining/val': curr_number,
                                         'CLFtraining/train': curr_number_ref},
                                        step=epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}% BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))

class gen_linear_clf(object):
    def __init__(self,
                 clf_net,
                 clf_opt,
                 gen,
                 train_loader,
                 test_loader,
                 device='cuda',
                 batch_size=1024,
                 num_epochs=50,
                 disable_tqdm=False,
                 writer=None,
                 ):

        self.clf = clf_net
        self.opt = clf_opt

        self.model = gen
        self.best_number = 0.0
        self.crit = torch.nn.CrossEntropyLoss()

        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.disable_tqdm = disable_tqdm
        self.writer = writer

        self.data_train = Simple_Trans(self.compute_rep(train_loader))
        self.data_train = DataLoader(self.data_train, batch_size=batch_size)
        self.data_test = Simple_Trans(self.compute_rep(test_loader))
        self.data_test = DataLoader(self.data_test, batch_size=batch_size)

        self.train_angle_layer()

        self.train_acc, _ = self.compute_angle_acc(self.data_train)
        self.test_acc, _ = self.compute_angle_acc(self.data_test)

    def compute_rep(self, dataloader):
        reps, labels = [], []
        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            labels.append(label[:, 0])

            # forward
            with torch.no_grad():
                _, representation = self.model(x)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        # self.net.train()
        return [reps, labels]

    def compute_angle_acc(self, dataloader):
        self.clf.eval()

        running_eval_loss = 0.0
        right, total = [], []
        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            with torch.no_grad():
                preds = self.clf(x)
            # compute accuracy
            loss = self.crit(preds, label)
            running_eval_loss += loss

            _, pred_class = torch.max(preds, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))

        self.clf.train()
        return sum(right) / sum(total), running_eval_loss

    def train_angle_layer(self):
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm, position=0, leave=True)
        for epoch in progress_bar:
            for x, label in self.data_train:
                self.clf.train()
                self.opt.zero_grad()

                x, label = x.to(self.device), label.to(self.device)
                # print(x.shape)
                preds = self.clf(x)
                loss = self.crit(preds, label)

                # backward
                loss.backward()
                self.opt.step()

            curr_number_ref, test_loss_train = self.compute_angle_acc(self.data_train)
            curr_number, test_loss_test = self.compute_angle_acc(self.data_test)
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer is not None:
                self.writer.log_metrics({'CLFtraining/val': curr_number,
                                         'CLFtraining/train': curr_number_ref},
                                        step=epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}% BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))

class transfer_mlp(object):
    def __init__(self, v, MLP, MLP_optim,
                 train_loader, test_loader,
                 logger, total_epoch):
        """with fixed v, train the MLP with MLP_optim
        the train_loader and test_loader are both for test direction sets"""
        self.v = v
        self.MLP = MLP
        self.MLP_optim = MLP_optim

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.logger = logger
        self.total_epoch = total_epoch

        self.crit = torch.nn.CrossEntropyLoss()
        # self.transform = Rearrange('b n t d -> (b t) (n d)')

        self.train_mlp()
        self.final_loss, self.final_perf = self.eval_mlp()

    def eval_mlp(self):
        running_eval_loss = 0.0
        self.MLP.eval()
        self.v.eval()
        with torch.no_grad():
            right, total = [], []
            for x, label in self.test_loader:
                x, label = x.cuda(), label.cuda()

                # if x.shape[1] == 2:
                #     x[:, 1, :] = x[:, 0, :]

                latents = self.v.latents(x)
                preds = self.MLP((latents))
                label = rearrange(label, 'b t -> (b t)')

                loss = self.crit(preds, label)
                running_eval_loss += loss

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))
        self.MLP.train()
        return running_eval_loss, sum(right) / sum(total)

    def train_mlp(self):

        progress_bar = tqdm(range(self.total_epoch), position=0, leave=True)
        for epoch in progress_bar:
            right, total = [], []

            for x, label in self.train_loader:
                self.v.eval()
                self.MLP.train()
                self.MLP_optim.zero_grad()
                x, label = x.cuda(), label.cuda()

                with torch.no_grad():
                    latents = self.v.latents(x)

                preds = self.MLP((latents))
                label = rearrange(label, 'b t -> (b t)')
                loss = self.crit(preds, label)

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))

                loss.backward()
                self.MLP_optim.step()

            self.logger.log_metrics({'Loss/train_clf': sum(right) / sum(total)}, step=epoch)

            running_eval_loss, eval_clf = self.eval_mlp()
            self.logger.log_metrics({'Loss/eval_clf': eval_clf}, step=epoch)
            self.logger.log_metrics({'Loss/running_eval_loss': running_eval_loss}, step=epoch)

            progress_bar.set_description('Loss/train_clf: {}, Loss/eval_clf: {}'.format(sum(right)/sum(total), eval_clf))
