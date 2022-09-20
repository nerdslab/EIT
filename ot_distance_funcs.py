import torch
import torch.nn as nn
import numpy as np

import ot
from tqdm import tqdm

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])

        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def compute(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss



def compute_wasserstein_distance(xs, xt):
    cost_matrix = ot.dist(xs, xt)
    n = xs.shape[0]
    m = xt.shape[0]
    mu = 1 / n * np.ones([n, 1])
    nu = 1 / m * np.ones([m, 1])
    ot_value = ot.emd2(mu.reshape(-1), nu.reshape(-1), cost_matrix)
    return ot_value

def compute_OT(Xs, Xt):
    # Xs is of shape (num_trials, num_neurons, num_feats)
    num_neurons_s = Xs.shape[1]
    num_neurons_t = Xt.shape[1]
    wasserstein_distance = np.zeros((num_neurons_s, num_neurons_t))

    for i in tqdm(range(num_neurons_s)):
        for j in range(num_neurons_t):
            wasserstein_distance[i, j] = compute_wasserstein_distance(Xs[:, i], Xt[:, j])
    return wasserstein_distance

def test():

    for N in [100, 200, 500, 1000, 3000]:
        # should be similar okay.
        x = torch.rand(200, 16) - 0.5
        y = torch.rand(200, 16) - 0.5

        Helper = MMD_loss()
        loss = Helper.compute(source=x, target=y)
        print(loss)
