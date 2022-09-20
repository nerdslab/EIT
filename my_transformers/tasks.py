import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import matplotlib.pyplot as plt


class Simple_Trans(Dataset):
    def __init__(self, data, transform=None):
        # [reps, labels]
        self.reps = data[0]
        self.labels = data[1]
        # print(self.reps.shape, self.labels.shape) # torch.Size([60000, 64]) torch.Size([60000])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.reps[idx, :], self.labels[idx]

class linear_clf(object):
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
                 trainit=True,
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

        if trainit:
            self.train_linear_layer()

            self.train_acc = self.compute_acc(self.data_train)
            self.test_acc = self.compute_acc(self.data_test)

    def compute_rep(self, dataloader):
        reps, labels = [], []
        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            #labels.append(label)
            labels.append(label)
            #labels.append(label)

            # forward
            with torch.no_grad():
                #_, representation, _ = self.model(x)
                #reps.append(representation.detach().cpu())
                #_, representation, _ = self.model(x)
                _, representation, _ = self.model(x)
                reps.append(representation.detach().cpu())
                #reps.append(representation["A"].detach().cpu())
                #reps.append(representation["B"].detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        # self.net.train()
        return [reps, labels]

    def compute_acc(self, dataloader):
        self.clf.eval()
        right = []
        total = []
        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_logits = self.clf(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))
        self.clf.train()
        # self.net.train()
        return sum(right) / sum(total)

    def train_linear_layer(self):
        class_criterion = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm, position=0, leave=True)
        for epoch in progress_bar:
            for x, label in self.data_train:
                self.clf.train()
                self.opt.zero_grad()

                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.clf(x)
                loss = class_criterion(pred_class, label)

                # backward
                loss.backward()
                self.opt.step()

            curr_number = self.compute_acc(self.data_test)
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer is not None:
                #self.writer.log_metrics({'CLFtraining/val': curr_number}, step=epoch)
                self.writer.add_scalar('CLFtraining/val', curr_number, global_step=epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}% BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))


def compute_acc(model, c_loader):
    model.eval()

    right = []
    total = []
    for data in c_loader:
        dataA = data['A'].cuda()
        dataB = data['B'].cuda()
        labelA = torch.zeros(dataA.shape[0]).long().cuda()
        labelB = torch.ones(dataB.shape[0]).long().cuda()
        with torch.no_grad():
            predA = model(dataA)
            predB = model(dataB)
        _, predA_class = torch.max(predA, 1)
        _, predB_class = torch.max(predB, 1)

        right.append((predA_class == labelA).sum().item())
        right.append((predB_class == labelB).sum().item())
        total.append(labelA.size(0))
        total.append(labelB.size(0))

    model.train()
    return sum(right) / sum(total)

def plot_construct(model, data):
    model.eval()

    dataA = data["A"].cuda()
    dataB = data["B"].cuda()

    recon = model(dataA)
    reconB = model(dataB)
    fig, axes = plt.subplots(nrows=4, ncols=6)
    for i in range(4):
        ori_i = dataA[i]
        ori_i_B = dataB[i]
        recon_i = recon[i]
        recon_i_B = reconB[i]

        ori_i = torch.permute(ori_i, (1, 2, 0)).detach().cpu()
        recon_i = torch.permute(recon_i, (1, 2, 0)).detach().cpu()
        axes[i, 2].imshow(recon_i)
        recon_i = torch.sigmoid(recon_i)

        axes[i, 0].imshow(ori_i)
        axes[i, 1].imshow(recon_i)

        ori_i_B = torch.permute(ori_i_B, (1, 2, 0)).detach().cpu()
        recon_i_B = torch.permute(recon_i_B, (1, 2, 0)).detach().cpu()
        axes[i, 5].imshow(recon_i_B)

        recon_i_B = torch.sigmoid(recon_i_B)
        axes[i, 3].imshow(ori_i_B)
        axes[i, 4].imshow(recon_i_B)

    model.train()
    return fig


"""just train a supervised clf to see if the network is working"""
def main(v, name, TOTAL_EPOCH, train_dataloader, test_dataloader):

    crit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(v.parameters(), lr=0.001)
    writer = SummaryWriter('runs/' + name)

    pbar = tqdm(range(TOTAL_EPOCH), position=0, leave=True)
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            dataA = data['A'].cuda()
            dataB = data['B'].cuda()
            labelA = torch.zeros(dataA.shape[0]).long().cuda()
            labelB = torch.ones(dataB.shape[0]).long().cuda()

            optimizer.zero_grad()
            predA = v(dataA)
            predB = v(dataB)

            lossA = crit(predA, labelA)
            lossB = crit(predB, labelB)
            loss = lossA + lossB
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            writer.add_scalar('training loss', running_loss, epoch * len(train_dataloader) + i)

        train_acc = compute_acc(v, train_dataloader)
        test_acc = compute_acc(v, test_dataloader)

        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        pbar.set_description("Current loss {:.3f}, train_acc {}, test_acc {}".format(running_loss, train_acc, test_acc))


"""reconstruction, on both datasets"""
def reconstruct(model, name, TOTAL_EPOCH, train_dataloader, test_dataloader):
    crit = nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0001)
    writer = SummaryWriter('runs/' + name)

    data_batch_train = next(iter(train_dataloader))
    data_batch_test = next(iter(test_dataloader))

    pbar = tqdm(range(TOTAL_EPOCH), position=0, leave=True)
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            """don't consider data B at first"""
            optimizer.zero_grad()

            dataA = data['A'].cuda()
            dataB = data['B'].cuda()

            reconA = model(dataA)
            reconA = torch.sigmoid(reconA)
            reconB = model(dataB)
            reconB = torch.sigmoid(reconB)

            lossA = crit(reconA, dataA)
            lossB = crit(reconB, dataB)
            loss = lossA + lossB

            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            writer.add_scalar('training loss', running_loss, epoch * len(train_dataloader) + i)

        writer.add_figure('reconstruction train', plot_construct(model, data_batch_train), global_step=epoch)
        writer.add_figure('reconstruction test', plot_construct(model, data_batch_test), global_step=epoch)

        pbar.set_description("Current loss {:.3f}".format(running_loss))
