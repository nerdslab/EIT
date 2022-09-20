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

class transfer_mlp_ssl(object):
    def __init__(self, v, s, MLP, MLP_optim,
                 train_loader, test_loader,
                 logger, total_epoch):
        """with fixed v, train the MLP with MLP_optim
        the train_loader and test_loader are both for test direction sets"""
        self.v = v
        self.s = s
        self.v.eval()
        self.s.eval()

        self.MLP = MLP
        self.MLP_optim = MLP_optim

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.logger = logger
        self.total_epoch = total_epoch

        self.crit = torch.nn.CrossEntropyLoss()

        self.train_mlp()
        self.final_loss, self.final_perf = self.eval_mlp()

    def eval_mlp(self):
        running_eval_loss = 0.0
        self.MLP.eval()
        with torch.no_grad():
            right, total = [], []
            for x, label in self.test_loader:
                x, label = x.cuda(), label.cuda()

                trans_x, small_trans_x = self.v.get_latent_t(x)
                latents, _ = self.s(trans_x, small_trans_x, ssl=False)

                preds = self.MLP(latents)
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
                self.s.eval()
                self.MLP.train()
                self.MLP_optim.zero_grad()
                x, label = x.cuda(), label.cuda()

                with torch.no_grad():
                    trans_x, small_trans_x = self.v.get_latent_t(x)
                    latents, _ = self.s(trans_x, small_trans_x, ssl=False)
                    # print(latents.shape)

                preds = self.MLP(latents)
                label = rearrange(label, 'b t -> (b t)')
                loss = self.crit(preds, label)

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))

                loss.backward()
                self.MLP_optim.step()

            progress_bar.set_description('Loss/train_clf: {}'.format(sum(right) / sum(total)))
            self.logger.log_metrics({'Loss_mlp/train_clf': sum(right) / sum(total)}, step=epoch)

            running_eval_loss, eval_clf = self.eval_mlp()
            self.logger.log_metrics({'Loss_mlp/eval_clf': eval_clf}, step=epoch)
            self.logger.log_metrics({'Loss_mlp/running_eval_loss': running_eval_loss}, step=epoch)


class transfer_bchmk(object):
    # logic:
    # train MLP_base on data_A_train_latents, test on data_A_test_latents
    # use latents produced by MLP_base, train a new MLP on data_B_train_latents, test on data_B_test_latents
    def __init__(self, MLP_base,
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
                       label_B_test,
                       total_epochs=200):

        self.crit = torch.nn.CrossEntropyLoss()
        train_A_loader, test_A_loader = self.generate_A_loader(data_A_train_latents, data_A_test_latents,
                                                               label_A_train, label_A_test)
        self.train_MLP(MLP_base, MLP_base_optim, train_A_loader, test_A_loader, total_epochs=total_epochs)

        train_B_loader, test_B_loader = self.generate_B_loader(data_B_train_latents, data_B_test_latents, MLP_base,
                                                               label_B_train, label_B_test)
        self.train_MLP(MLP_trans, MLP_trans_optim, train_B_loader, test_B_loader, reshape=False, total_epochs=total_epochs)

    def train_MLP(self, MLP, MLP_optim, train_loader, test_loader, reshape=True, total_epochs=50):
        progress_bar = tqdm(range(total_epochs), position=0, leave=True)
        for epoch in progress_bar:
            right, total = [], []

            for x, label in train_loader:
                MLP.train()
                MLP_optim.zero_grad()
                x, label = x.cuda(), label.cuda()

                preds = MLP(x.float())
                if reshape:
                    label = rearrange(label, 'b t -> (b t)')
                loss = self.crit(preds, label)

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))

                loss.backward()
                MLP_optim.step()

            running_eval_loss, eval_clf = self.eval_MLP(MLP, test_loader, reshape=reshape)

            progress_bar.set_description(
                'Loss/train_clf: {}, Loss/eval_clf: {}'.format(sum(right) / sum(total), eval_clf))

    def eval_MLP(self, MLP, test_loader, reshape):
        running_eval_loss = 0.0
        MLP.eval()
        with torch.no_grad():
            right, total = [], []
            for x, label in test_loader:
                x, label = x.cuda(), label.cuda()

                preds = MLP(x.float())
                if reshape:
                    label = rearrange(label, 'b t -> (b t)')

                loss = self.crit(preds, label)
                running_eval_loss += loss

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))
        MLP.train()
        return running_eval_loss, sum(right) / sum(total)

    def generate_A_loader(self, data_A_train_latents, data_A_test_latents, label_A_train, label_A_test):
        data_A_train_latents = torch.from_numpy(data_A_train_latents).float()
        data_A_test_latents = torch.from_numpy(data_A_test_latents).float()

        data_A_train = Simple_Trans([data_A_train_latents, label_A_train])
        data_A_test = Simple_Trans([data_A_test_latents, label_A_test])
        return DataLoader(data_A_train, batch_size=64, shuffle=True),\
               DataLoader(data_A_test, batch_size=64)

    def generate_B_loader(self, data_B_train_latents, data_B_test_latents, MLP_base, label_B_train, label_B_test):
        data_B_train_latents = torch.from_numpy(data_B_train_latents).float()
        data_B_test_latents = torch.from_numpy(data_B_test_latents).float()

        MLP_base.eval()
        with torch.no_grad():
            data_B_train_latents, data_B_test_latents = data_B_train_latents.cuda(), data_B_test_latents.cuda()
            data_B_train_data = MLP_base.latents(data_B_train_latents)
            data_B_test_data = MLP_base.latents(data_B_test_latents)

        label_B_train = rearrange(label_B_train, 'b t -> (b t)')
        label_B_test = rearrange(label_B_test, 'b t -> (b t)')

        data_B_train = Simple_Trans([data_B_train_data, label_B_train])
        data_B_test = Simple_Trans([data_B_test_data, label_B_test])
        return DataLoader(data_B_train, batch_size=64, shuffle=True),\
               DataLoader(data_B_test, batch_size=64)

class transfer_bdt_bchmk(object):
    # logic:
    # train MLP_base on data_A_train_latents, test on data_A_test_latents
    # use latents produced by MLP_base, train a new MLP on data_B_train_latents, test on data_B_test_latents
    def __init__(self, MLP_base,
                       MLP_base_optim,
                       MLP_trans,
                       MLP_trans_optim,
                       train_A_loader,
                       test_A_loader,
                       train_B_loader,
                       test_B_loader,
                       total_epochs=200):

        self.crit = torch.nn.CrossEntropyLoss()

        # normally train
        self.train_MLP(MLP_base, MLP_base_optim, train_A_loader, test_A_loader, total_epochs=total_epochs)
        self.train_new_end(MLP_trans, MLP_base, MLP_trans_optim, train_B_loader, test_B_loader, reshape=True, total_epochs=total_epochs)

    def train_MLP(self, MLP, MLP_optim, train_loader, test_loader, reshape=True, total_epochs=50):
        progress_bar = tqdm(range(total_epochs), position=0, leave=True)
        for epoch in progress_bar:
            right, total = [], []

            for x, label in train_loader:
                MLP.train()
                MLP_optim.zero_grad()
                x, label = x.cuda(), label.cuda()

                preds = MLP(x.float())
                if reshape:
                    label = rearrange(label, 'b t -> (b t)')
                loss = self.crit(preds, label)

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))

                loss.backward()
                MLP_optim.step()

            running_eval_loss, eval_clf = self.eval_MLP(MLP, test_loader, reshape=reshape)

            progress_bar.set_description(
                'Loss/train_clf: {}, Loss/eval_clf: {}'.format(sum(right) / sum(total), eval_clf))

    def eval_MLP(self, MLP, test_loader, reshape):
        running_eval_loss = 0.0
        MLP.eval()
        with torch.no_grad():
            right, total = [], []
            for x, label in test_loader:
                x, label = x.cuda(), label.cuda()

                preds = MLP(x.float())
                if reshape:
                    label = rearrange(label, 'b t -> (b t)')

                loss = self.crit(preds, label)
                running_eval_loss += loss

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))
        MLP.train()
        return running_eval_loss, sum(right) / sum(total)

    def train_new_end(self, MLP_trans, MLP_base, MLP_optim, train_loader, test_loader, reshape=True, total_epochs=50):
        progress_bar = tqdm(range(total_epochs), position=0, leave=True)
        for epoch in progress_bar:
            right, total = [], []

            for x, label in train_loader:
                MLP_trans.train()
                MLP_optim.zero_grad()
                x, label = x.cuda(), label.cuda()

                preds_mid = MLP_trans(x.float())
                MLP_base.eval()
                preds = MLP_base.to_end(preds_mid.clone())
                if reshape:
                    label = rearrange(label, 'b t -> (b t)')
                loss = self.crit(preds, label)

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))

                loss.backward()
                MLP_optim.step()

            running_eval_loss, eval_clf = self.eval_new_MLP(MLP_trans, MLP_base, test_loader, reshape=reshape)

            progress_bar.set_description(
                'Loss/train_clf: {}, Loss/eval_clf: {}'.format(sum(right) / sum(total), eval_clf))

    def eval_new_MLP(self, MLP_trans, MLP_base, test_loader, reshape):
        running_eval_loss = 0.0
        MLP_trans.eval()
        MLP_base.eval()
        with torch.no_grad():
            right, total = [], []
            for x, label in test_loader:
                x, label = x.cuda(), label.cuda()

                preds = MLP_base.to_end(MLP_trans(x.float()))
                if reshape:
                    label = rearrange(label, 'b t -> (b t)')

                loss = self.crit(preds, label)
                running_eval_loss += loss

                _, pred_class = torch.max(preds, 1)
                right.append((pred_class == label).sum().item())
                total.append(label.size(0))
        MLP_trans.train()
        return running_eval_loss, sum(right) / sum(total)
