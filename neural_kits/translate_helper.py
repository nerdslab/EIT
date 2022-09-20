import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from einops import rearrange, repeat

class latent_dataset(Dataset):
    def __init__(self, latents_bf, latents_af):
        self.data_bf = latents_bf[0]
        self.data_af = latents_af[0]

        # print(self.data_bf.shape, self.data_af.shape)
        # torch.Size([2004, 163, 16]) torch.Size([2004, 163, 1])

    def __getitem__(self, index):
        return self.data_bf[index], self.data_af[index]

    def __len__(self):
        return self.data_bf.shape[0]

class latent_label_dataset(Dataset):
    def __init__(self, latents):
        self.data = latents[0][:, :]
        self.label = rearrange(latents[1], 'b t -> (b t)')

        # print(self.data.shape, self.label.shape)
        # torch.Size([2064, 152, 16]) torch.Size([1032, 2])

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


class translate_helper_H(object):
    def __init__(self, trans_model, trans_optim, trans_MLP, MLP_optim):
        self.trans_model = trans_model
        self.trans_optim = trans_optim
        self.trans_MLP = trans_MLP
        self.MLP_optim = MLP_optim

    def train_translate(self, T_A_train, S_A_train, T_A_test, S_A_test, total_epoch=200):
        crit = nn.L1Loss()
        train_data = latent_dataset(T_A_train, S_A_train)
        train_loader = DataLoader(train_data, batch_size=64)

        test_data = latent_dataset(T_A_test, S_A_test)
        test_loader = DataLoader(test_data, batch_size=64)

        progress_bar = tqdm(range(total_epoch), position=0, leave=True)
        for epoch in progress_bar:
            running_train_loss = 0.
            running_test_loss = 0.

            for bf, af in train_loader:
                self.trans_model.train()
                self.trans_optim.zero_grad()

                bf, af = bf.cuda(), af.cuda()
                # print(bf.shape, af.shape)
                pred_af = self.trans_model(bf)

                loss = crit(pred_af, af)
                running_train_loss += loss
                loss.backward()
                self.trans_optim.step()

            for bf, af in test_loader:
                self.trans_model.eval()
                with torch.no_grad():
                    bf, af = bf.cuda(), af.cuda()
                    pred_af = self.trans_model(bf)
                    loss = crit(pred_af, af)
                    running_test_loss += loss

            progress_bar.set_description('Loss/train: {}; Loss/test: {}'.format(running_train_loss, running_test_loss))

    def translate_MLP(self, T_B_train, T_B_test, total_epoch=20):
        crit = nn.CrossEntropyLoss()

        train_data = latent_label_dataset(T_B_train)
        train_loader = DataLoader(train_data, batch_size=64)

        test_data = latent_label_dataset(T_B_test)
        test_loader = DataLoader(test_data, batch_size=64)

        self.trans_model.eval()
        progress_bar = tqdm(range(total_epoch), position=0, leave=True)

        right_train, total_train = [], []
        for epoch in progress_bar:
            running_train_loss = 0.
            running_test_loss = 0.

            for latent_bf, label in train_loader:
                latent_bf, label = latent_bf.cuda(), label.cuda()
                b, n, dim = latent_bf.shape
                with torch.no_grad():
                    latent_af = self.trans_model(latent_bf)
                latent_af = rearrange(latent_af, 'b n dim -> b (n dim)')

                self.trans_MLP.train()
                self.MLP_optim.zero_grad()

                preds = self.trans_MLP(latent_af)

                loss = crit(preds, label)
                running_train_loss += loss
                loss.backward()
                self.trans_optim.step()

                _, pred_class = torch.max(preds, 1)
                right_train.append((pred_class == label).sum().item())
                total_train.append(label.size(0))

            right_eval, total_eval = [], []
            for latent_bf, label in train_loader:
                latent_bf, label = latent_bf.cuda(), label.cuda()
                b, n, dim = latent_bf.shape
                with torch.no_grad():
                    latent_af = self.trans_model(latent_bf)
                latent_af = rearrange(latent_af, 'b n dim -> b (n dim)')

                self.trans_MLP.eval()
                with torch.no_grad():
                    preds = self.trans_MLP(latent_af)
                    loss = crit(preds, label)
                    running_test_loss += loss

                _, pred_class = torch.max(preds, 1)
                right_eval.append((pred_class == label).sum().item())
                total_eval.append(label.size(0))

            progress_bar.set_description('Loss/train: {}; Loss/test: {}; Clf/train: {}; Clf/test: {}'.format(running_train_loss,
                                                                                                             running_test_loss,
                                                                                                             sum(right_train)/sum(total_train),
                                                                                                             sum(right_eval)/sum(total_eval)))

    def test_translate_MLP(self, T_B_test):
        ...
