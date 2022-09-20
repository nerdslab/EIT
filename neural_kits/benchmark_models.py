import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from my_transformers.model_utils import *
from einops import repeat, rearrange


def l2(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return (2 - 2 * (x * y).sum(dim=-1))

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def reconstruction_loss(x, x_recon, distribution='poisson'):
    '''
        VAE works the best with bernoulli loss
        i-VAE works the best with poisson loss
    '''
    batch_size = x.size(0) # [256 B, 163]
    assert batch_size != 0

    if distribution == 'bernoulli': #
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'weighted_bernoulli':
         weight = torch.tensor([0.1, 0.9]).to("cuda") # just a label here
         weight_ = torch.ones(x.shape).to("cuda")
         weight_[x <= 0.5] = weight[0]
         weight_[x > 0.5] = weight[1]
         recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none')
         # print(recon_loss.shape) # torch.Size([256, 163])
         recon_loss = torch.sum(weight_ * recon_loss).div(batch_size)

    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'poisson':
        # print((x - x_recon * torch.log(x)).shape)
        #print(x_recon)
        x_recon.clamp(min=1e-7, max=1e7)
        recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class Neural_MLP(nn.Module):
    def __init__(self, neuron, time=8, dropout=0.2, latent=128, cls=8):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(neuron*time, latent*time),
            nn.BatchNorm1d(latent*time),
            nn.Dropout(dropout)
        )

        self.final = nn.Sequential(
            nn.Linear(latent, cls),
        )
        mode = 'no-test'
        if mode == 'test':
            self.linear = nn.Sequential(
                nn.Linear(neuron*time, latent * time),
                nn.BatchNorm1d(latent*time),
                nn.Dropout(dropout),
                nn.Linear(latent * time, latent * time),
                nn.BatchNorm1d(latent * time),
                nn.Dropout(dropout),
            )

            self.final = nn.Sequential(
                nn.Linear(latent, cls),
            )


        self.time=time
        self.cls=cls

    def latents(self, img):
        b, t, n = img.shape
        x = rearrange(img, 'b t n -> b (t n)')
        x = self.linear(x)  # b (t l)
        x = rearrange(x, 'b (t l) -> (b t) l', t=self.time)
        return x

    def forward(self, img):
        b, t, n = img.shape

        x = rearrange(img, 'b t n -> b (t n)')
        x = self.linear(x) # b (t l)
        x = rearrange(x, 'b (t l) -> (b t) l', t=self.time)
        x = self.final(x)

        return x, None

class Neural_GRU(nn.Module):
    """neuron amount independent, dynamic learner"""
    def __init__(self, *,
                 num_classes=8,  # amount of final classification cls
                 neuron=160,
                 dropout_p=0.2,
                 hidden_size=256,
                 ):
        super().__init__()
        self.hidden_size = hidden_size

        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes
        self.encoder = nn.GRU(
            input_size=neuron,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def latents(self, img):
        bs, t, num_n = img.size()
        s_n, h_n = self.encoder(img)
        s_n = self.dropout(s_n)  # dropout
        s_n = s_n.reshape(bs * t, self.hidden_size*2)
        return s_n

    def forward(self, img):
        bs, t, num_n = img.size()
        s_n, h_n = self.encoder(img)
        s_n = self.dropout(s_n) #dropout
        s_n = s_n.reshape(bs*t, self.hidden_size*2)
        fc = self.fc(s_n)
        return fc, None

class Neural_beta(nn.Module):
    def __init__(self, neuron, time=8, beta=2,
                 l_dim=16*8, hidden_dim = [128], batchnorm=False):
        super().__init__()
        self.beta = beta

        self.l_dim = l_dim
        self.layers_dim = [neuron*time, *hidden_dim]  # [163, (163, 128)]

        e_modules = []
        e_modules.append(nn.Dropout(0.2))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            e_modules.append(nn.Linear(in_dim, out_dim))
            if batchnorm:
                e_modules.append(nn.BatchNorm1d(num_features=out_dim))
            e_modules.append(nn.ReLU(True))
        e_modules.append(nn.Linear(self.layers_dim[-1], 2 * self.l_dim))
        self.encoder = nn.Sequential(*e_modules)

        self.layers_dim.reverse()
        d_modules = []
        d_modules.append(nn.Linear(self.l_dim, self.layers_dim[0]))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            if batchnorm:
                d_modules.append(nn.BatchNorm1d(num_features=in_dim))
            d_modules.append(nn.ReLU(True))
            d_modules.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*d_modules)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, img):
        b, t, n = img.shape

        img = rearrange(img, 'b t n -> b (t n)')
        distributions = self.encoder(img)
        mu = distributions[:, :self.l_dim]
        logvar = distributions[:, self.l_dim:]
        z = reparametrize(mu, logvar)

        x_recon = (self.decoder(z).view(img.size()))

        recon_loss = reconstruction_loss(img, x_recon, distribution='poisson')
        kl_loss, _, _ = kl_divergence(mu, logvar)

        return recon_loss+self.beta*kl_loss, mu

class Neural_swap(nn.Module):
    def __init__(self, neuron, time=8, beta=2, alpha=1,
                 l_dim=16*8, hidden_dim = [128], batchnorm=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.s_dim = round(l_dim/2)
        self.l_dim = l_dim
        self.c_dim = int(l_dim - self.s_dim)

        self.layers_dim = [neuron*time, *hidden_dim]  # [163, (163, 128)]

        e_modules = []
        # e_modules.append(nn.Dropout(0.2))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            e_modules.append(nn.Linear(in_dim, out_dim))
            if batchnorm:
                e_modules.append(nn.BatchNorm1d(num_features=out_dim))
            e_modules.append(nn.ReLU(True))
        e_modules.append(nn.Linear(self.layers_dim[-1], self.l_dim+self.s_dim))
        self.encoder = nn.Sequential(*e_modules)

        self.layers_dim.reverse()
        d_modules = []
        d_modules.append(nn.Linear(self.l_dim, self.layers_dim[0]))
        for in_dim, out_dim in zip(self.layers_dim[:-1], self.layers_dim[1:]):
            if batchnorm:
                d_modules.append(nn.BatchNorm1d(num_features=in_dim))
            d_modules.append(nn.ReLU(True))
            d_modules.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*d_modules)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def augment(self, x):
        transform = Compose(RandomizedDropout(p=0.2, apply_p=1),
                                       Pepper(p=0.5, sigma=1.0, apply_p=0.2),)
        x1 = transform(x)
        x2 = transform(x)
        return x1, x2

    def forward(self, x):
        x = rearrange(x, 'b t n -> b (t n)')
        x1, x2 = self.augment(x)

        distributions_ref = self.encoder(x)
        mu_ref = distributions_ref[:, :self.l_dim]

        # x1 = rearrange(x1, 'b t n -> b (t n)')
        # x2 = rearrange(x2, 'b t n -> b (t n)')

        # get c and s for x1
        distributions1 = self.encoder(x1)
        c1 = distributions1[:, :self.c_dim]
        mu1 = distributions1[:, self.c_dim:self.l_dim]
        logvar1 = distributions1[:, self.l_dim:]
        s1 = reparametrize(mu1, logvar1)

        # get c and s for x2
        distributions2 = self.encoder(x2)
        c2 = distributions2[:, :self.c_dim]
        mu2 = distributions2[:, self.c_dim:self.l_dim]
        logvar2 = distributions2[:, self.l_dim:]
        s2 = reparametrize(mu2, logvar2)

        # create new z1 and z2 by exchanging the content
        z1_new = torch.cat([c2, s1], dim=1)
        z2_new = torch.cat([c1, s2], dim=1)

        #### exchange content reconsturction
        x1_recon = (self.decoder(z1_new).view(x1.size()))
        x2_recon = (self.decoder(z2_new).view(x1.size()))

        #### original reconstruction
        z1_ori = torch.cat([c1, s1], dim=1)
        z2_ori = torch.cat([c2, s2], dim=1)
        x1_recon_ori = (self.decoder(z1_ori).view(x1.size()))
        x2_recon_ori = (self.decoder(z2_ori).view(x1.size()))

        recon1 = reconstruction_loss(x1, x1_recon, distribution="poisson")
        recon2 = reconstruction_loss(x2, x2_recon, distribution="poisson")
        recon1_ori = reconstruction_loss(x1, x1_recon_ori, distribution="poisson")
        recon2_ori = reconstruction_loss(x2, x2_recon_ori, distribution="poisson")
        kl1, _, _ = kl_divergence(mu1, logvar1)
        kl2, _, _ = kl_divergence(mu2, logvar2)

        l2_loss = l2(c1, c2).mean()
        return l2_loss + self.alpha*(recon1 + recon2 + recon1_ori + recon2_ori)/2 + self.beta*(kl1 + kl2)/2, mu_ref

class Transfer_MLP(nn.Module):
    def __init__(self, neuron, latent=12, dropout=0.2, cls=8):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent, 2),
        )

        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(neuron, cls),
        )

        self.cls = cls

    def latents(self, img):
        b, t, n = img.shape
        x = rearrange(img, 'b t n -> b n t')
        x = self.linear(x)  # b (t l)
        x = rearrange(x, 'b n t -> (b t) n')
        return x

    def forward(self, img):
        b, t, n = img.shape

        x = rearrange(img, 'b t n -> b n t')
        x = self.linear(x) # b (t l)
        x = rearrange(x, 'b n t -> (b t) n')
        x = self.final(x)

        return x

class Transfer_MLP_new_end(nn.Module):
    def __init__(self, neuron_new, dropout=0.2, cls=8):
        super().__init__()
        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(neuron_new, cls),)

    def forward(self, latent):
        "latent shape is [b n t]"
        # latent = rearrange(latent, 'b n t -> b t n')
        return self.final(latent)

class Transfer_NDT(nn.Module):
    def __init__(self, *,
                 neuron=160,
                 num_classes,  # amount of final classification cls
                 depth=4,  # vit depth
                 heads=6,  # vit heads
                 mlp_expend=2,  # vit mlp dim (typically 4 times expand in vision, here 2 is better)
                 dim_head=64,  # default is 64, better than 16/32/96
                 dropout=0.5,  # adjust from 0.2 to 0.5 -- performance varies
                 final_dim=None,
                 ):
        super().__init__()
        if final_dim == None:
            final_dim = neuron
        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes

        # neuron --> dim
        self.NDT_linear = nn.Linear(neuron, 2*neuron)
        self.NDT_transformer = Transformer(2*neuron,
                                           depth,
                                           heads,
                                           dim_head,
                                           2*neuron*mlp_expend,
                                           dropout,
                                           offset=False)
        self.NDT_temp_embed = nn.Parameter(torch.randn(1, 6, 2*neuron))

        self.NDT_back = nn.Linear(2*neuron, final_dim)
        self.NDT_cls = nn.Linear(final_dim, num_classes)

    def to_end(self, x):
        temp_token = repeat(self.NDT_temp_embed, '() t d -> b t d', b=x.shape[0])
        x = x + temp_token

        x, weights = self.NDT_transformer(x)  # [b t dim]
        x = self.NDT_back(x)
        x = self.NDT_cls(x)  # [b t cls]
        return rearrange(x, 'b t cls -> (b t) cls')

    def forward(self, img):
        b, t, n = img.shape

        x = self.NDT_linear(img)  # b t dim
        temp_token = repeat(self.NDT_temp_embed, '() t d -> b t d', b=b)
        x = x + temp_token

        x, weights = self.NDT_transformer(x)  # [b t dim]
        x = self.NDT_back(x)
        x = self.NDT_cls(x)  # [b t cls]

        return rearrange(x, 'b t cls -> (b t) cls')

class Transfer_NDT_new_end(nn.Module):
    def __init__(self, neuron_new, latents, dropout=0.2):
        super().__init__()
        self.new_linear = nn.Linear(neuron_new, latents)

    def forward(self, img):
        return self.new_linear(img)



class RandomizedDropout:
    def __init__(self, p: float = 0.5, apply_p=1.):
        self.p = p
        self.apply_p = apply_p

    def __call__(self, x):
        # generate a random dropout probability for each sample
        p = torch.rand(x.shape) * self.p
        # generate dropout mask
        dropout_mask = torch.rand(x.shape) < 1 - p
        # generate mask for applying dropout
        apply_mask = torch.rand(x.shape) < 1 - self.apply_p
        dropout_mask = dropout_mask + apply_mask
        r_x = x * dropout_mask.to(x)
        return r_x

class Pepper:
    def __init__(self, p=0.5, sigma=1.0, apply_p=1.):
        self.p = p
        self.sigma = sigma
        self.apply_p = apply_p

    def __call__(self, x):
        keep_mask = torch.rand(x.shape) < self.p
        random_pepper = self.sigma * keep_mask
        apply_mask = torch.rand(x.shape) < self.apply_p
        random_pepper = random_pepper * apply_mask
        return x + random_pepper.to(x)

class Compose:
    r"""Composes several transforms together.
    Args:
        transforms (Callable): List of transforms to compose.
    """
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            # print(x.shape)
            x = transform(x)
            # print(transform, x.shape)
        return x