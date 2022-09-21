from einops.layers.torch import Rearrange
from my_transformers.model_utils import *
import torch.nn.functional as F


neuron_amount = {'chewie1': 163,
                 'chewie2': 148,
                 'mihi1': 163,
                 'mihi2': 152,}

class Neural_ViT_T(nn.Module):
    """neuron amount independent, dynamic learner"""
    def __init__(self, *,
                 num_classes,  # amount of final classification cls
                 single_dim, # single neuron dim
                 depth,  # vit depth
                 heads,  # vit heads
                 neuron=160,
                 mlp_expend=2,  # vit mlp dim (typically 4 times expand in vision, here 2 is better)
                 dim_head=64,  # default is 64, better than 16/32/96
                 dropout=0,
                 neuron_dropout=0.,
                 ):
        super().__init__()
        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes
        self.bottelneck_dim = 1

        self.to_temp = Rearrange('b n t -> (b n) t')
        self.single_embed = nn.Linear(1, single_dim)
        self.T_transformer = Transformer(single_dim, depth, heads, dim_head, mlp_expend*single_dim, dropout,
                                         offset=True, ff=True)

        self.mlp_head_bottolneck = nn.Sequential(nn.Linear(single_dim, self.bottelneck_dim))  # 8*16

        # bottom mlp placeholder. Should be replaced with flexible bottom mlp soon
        self.mlp_head_bottom = nn.Sequential(
            nn.Dropout(neuron_dropout),
            #nn.Linear(neuron*self.bottelneck_dim, single_dim),
            #nn.Linear(single_dim, num_classes),
            nn.Linear(neuron*self.bottelneck_dim, num_classes), # for some reason above is synthetic, this is real
        )

    def bottom_head_within_cls(self):
        self.bottom_chewie1 = nn.Sequential(nn.Linear(neuron_amount['chewie1'] * self.bottelneck_dim, self.num_classes))
        self.bottom_chewie2 = nn.Sequential(nn.Linear(neuron_amount['chewie2'] * self.bottelneck_dim, self.num_classes))
        self.bottom_mihi1 = nn.Sequential(nn.Linear(neuron_amount['mihi1'] * self.bottelneck_dim, self.num_classes))
        self.bottom_mihi2 = nn.Sequential(nn.Linear(neuron_amount['mihi2'] * self.bottelneck_dim, self.num_classes))

        self.bottom_lookup = {'chewie1': self.bottom_chewie1,
                              'chewie2': self.bottom_chewie2,
                              'mihi1': self.bottom_mihi1,
                              'mihi2': self.bottom_mihi2,}

    def get_latent_t(self, img):
        b, t, n = img.shape

        # feed into t transformer, without temperol id
        img = torch.transpose(img, -1, -2)  # [batch, neuron, time]
        x = self.to_temp(img)[:, :, None]  # [(batch, neuron), time]
        trans_x = self.single_embed(x)  # [(batch, neuron), time, dim]
        trans_x, t_weights = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]

        trans_x = rearrange(trans_x, '(b n) t d -> b n t d', b=b)  # [batch, neuron, (time, dim)]
        small_trans_x = self.mlp_head_bottolneck(trans_x)
        return trans_x, small_trans_x

    def latents(self, img):
        b, t, n = img.shape

        # feed into t transformer, without temperol id
        img = torch.transpose(img, -1, -2)  # [batch, neuron, time]
        x = self.to_temp(img)[:, :, None]  # [(batch, neuron), time]
        trans_x = self.single_embed(x)  # [(batch, neuron), time, dim]
        trans_x, t_weights = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]

        trans_x = rearrange(trans_x, '(b n) t d -> b n t d', b=b)  # [batch, neuron, (time, dim)]
        small_trans_x = self.mlp_head_bottolneck(trans_x) # [batch, neuron, time 1]

        return rearrange(small_trans_x, 'b n t d -> (b t) (n d)')

    def translate_l(self, img):
        b, t, n = img.shape

        # feed into t transformer, without temperol id
        img = torch.transpose(img, -1, -2)  # [batch, neuron, time]
        x = self.to_temp(img)[:, :, None]  # [(batch, neuron), time]
        trans_x = self.single_embed(x)  # [(batch, neuron), time, dim]
        trans_x, t_weights = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]

        trans_x = rearrange(trans_x, '(b n) t d -> b n t d', b=b)  # [batch, neuron, (time, dim)]
        return rearrange(trans_x, 'b n t d -> (b t) n d')

    def forward(self, img, head=None, test=False):
        if test is False:
            if head is None:
                # use the pre-defined bottom with 160 neurons
                trans_x, small_trans_x = self.get_latent_t(img)
                x = rearrange(small_trans_x, 'b n t d -> b t (n d)')
                return rearrange(self.mlp_head_bottom(x), 'b t cls -> (b t) cls'), x
            elif type(head) is type('mihi1'):
                # use the self-stored bottom for different animals
                trans_x, small_trans_x = self.get_latent_t(img)
                x = rearrange(small_trans_x, 'b n t d -> b t (n d)')
                bottom = self.bottom_lookup[head]
                return rearrange(bottom(x), 'b t cls -> (b t) cls'), x
            else:
                # use the given bottom
                trans_x, small_trans_x = self.get_latent_t(img)
                x = rearrange(small_trans_x, 'b n t d -> b t (n d)')
                return rearrange(head(x), 'b t cls -> (b t) cls'), x

        else:
            trans_x, small_trans_x = self.get_latent_t(img)
            x = rearrange(small_trans_x, 'b n t d -> b t (n d)')
            small_trans_x = rearrange(self.mlp_head_bottom(x), 'b t cls -> (b t) cls')
            return trans_x, small_trans_x


class Neural_ViT_S(nn.Module):
    """takes a T_trans and do connectivity learning"""
    def __init__(self,
                 MT,
                 neuron,
                 num_classes,
                 single_dim,
                 embed_dim,
                 depth,
                 heads,
                 mlp_expend=1,
                 dim_head=64,
                 dropout=0.5,
                 emb_dropout=0.,
                 neuron_dropout=0.,
                 pool='cls',
                 type='cat',
                 offset=False,
                 ff=True,
                 ssl=False,
                 ):
        super().__init__()
        self.MT = MT
        self.bottelneck_dim = 1
        self.ssl = ssl
        if ssl is True:
            print("S_transformer is in SSL mode")
            self.decoder_pos_emb = nn.Parameter(torch.randn(single_dim))

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # assert time * single_dim == embed_dim, 'embed dim different from time*neuron-dim not implemented'

        '''three possible types: cls, cat, or time (else)
        mainly cls and cat are used'''
        self.type = type

        if self.type == 'cat':
            self.neuron_embed = nn.Parameter(torch.randn(1, neuron, single_dim))  # based on total amount of neuron
            self.dropout = nn.Dropout(neuron_dropout)
            self.S_transformer = Transformer(single_dim, depth, heads, dim_head,
                                             mlp_expend*single_dim, dropout,
                                             offset=offset, ff=ff)

            self.bottolneck = nn.Sequential(nn.Linear(single_dim, self.bottelneck_dim))
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(neuron*self.bottelneck_dim),
                nn.Dropout(neuron_dropout),
                nn.Linear(neuron*self.bottelneck_dim, num_classes)
            )

        elif self.type == 'cls':
            self.neuron_embed = nn.Parameter(torch.randn(1, neuron, single_dim))  # based on total amount of neuron
            self.cls_token = nn.Parameter(torch.randn(1, 1, single_dim))
            self.dropout = nn.Dropout(neuron_dropout)
            self.S_transformer = Transformer(single_dim, depth, heads, 32,
                                             mlp_expend*single_dim, dropout,
                                             offset=offset, ff=ff)

            self.pool = pool
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(single_dim),
                nn.Linear(single_dim, num_classes)
            )

        else:
            # In the benchmark model, pos embedding works the same as the neuron embedding
            self.neuron_embed = nn.Parameter(torch.randn(1, neuron, embed_dim))  # based on total amount of neuron
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.dropout = nn.Dropout(emb_dropout)
            self.S_transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_expend * embed_dim, dropout)

            self.pool = pool
            self.mlp_head = nn.Sequential(
                nn.LayerNorm((163+1)*2),
                nn.Linear((163+1)*2, num_classes)
            )

    def supervised_forward(self, img):
        b, t, n = img.shape
        trans_x, small_trans_x = self.MT(img, test=True)

        if self.type == 'cat':
            trans_x = rearrange(trans_x, 'b n t d -> (b t) n d')
            embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b * t)  # [(b t) n ed]

            x = trans_x + embed_x
            # x = self.dropout(x)
            x, weights = self.S_transformer(x)  # [(b t) n ed+td]

            x = self.bottolneck(x)  # [(b t) n 2]
            x = rearrange(x, 'b n d -> b (n d)')
            x = self.mlp_head(x)

        elif self.type == 'cls':
            trans_x = rearrange(trans_x, 'b n t d -> (b t) n d')
            embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b * t)  # [(b t) n ed]
            cls_token = repeat(self.cls_token, '() c d -> b c d', b=b * t)

            # trans_x = self.dropout(trans_x)
            x = trans_x + embed_x  # [(b t) n ed+td]
            x = self.dropout(x)  # change position of this?

            x = torch.cat([cls_token, x], dim=-2)  # [(b t) n+1 ed+td]
            x, weights = self.S_transformer(x)  # [(b t) n+1 ed+td]

            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.mlp_head(x)

        else:
            trans_x = rearrange(small_trans_x, 'b n t d -> b n (t d)')
            # embed with neuron id and feed it into spatial transformer
            embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b)  # [b n ed]
            cls_token = repeat(self.cls_token, '() 1 d -> b 1 d', b=b)

            x = trans_x + embed_x
            x = torch.cat([cls_token, x], dim=-2)  # [b n+1 ed+td]
            x = self.dropout(x)
            x, weights = self.S_transformer(x)  # [b n+1 ed+td]

            # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = rearrange(x, 'b n (t d) -> b t (n d)', t=t)
            x = self.mlp_head(x)
            x = rearrange(x, 'b t cls -> (b t) cls')

        # small_trans_x = rearrange(small_trans_x, 'b n t d -> (b t) (n d)')

        return {'S': x, 'T': small_trans_x}, {"weights": weights}

    def ssl_forward(self, img):
        trans_x, small_trans_x = self.MT.get_latent_t(img)  # trans_x shape [b n t d]
        trans_x = rearrange(trans_x, 'b n t d -> (b t) n d')

        bt, n, d = trans_x.shape
        device = trans_x.device

        # create masked_x and unmasked_x
        num_masked = int(self.masking_ratio * n)
        rand_indices = torch.rand(bt, n, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(bt, device=device)[:, None]
        unmasked_x = trans_x[batch_range, unmasked_indices]
        masked_x = trans_x[batch_range, masked_indices]

        # add unmask_x and masked_tokens with embeddings
        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=bt)
        unmasked_embed_x = embed_x[batch_range, unmasked_indices]
        masked_embed_x = embed_x[batch_range, masked_indices]

        masked_tokens = repeat(self.mask_token, 'd -> b n d', b=bt, n=num_masked)
        masked_tokens = masked_tokens + masked_embed_x
        unmasked_x = unmasked_x + unmasked_embed_x

        tokens = torch.cat((masked_tokens, unmasked_x), dim=1)
        x, weights = self.S_transformer(tokens)

        pred_masked_values = x[:, :num_masked]
        recon_loss = F.mse_loss(pred_masked_values, masked_x)
        return recon_loss

    def latents(self, img):

        b, t, n = img.shape
        trans_x, small_trans_x = self.MT.get_latent_t(img)

        assert self.type == 'cat'
        trans_x = rearrange(trans_x, 'b n t d -> (b t) n d')
        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b * t)  # [(b t) n ed]

        x = trans_x + embed_x
        x = self.dropout(x)
        x, weights = self.S_transformer(x)  # [(b t) n ed+td]

        x = self.bottolneck(x)  # [(b t) n 2]
        x = rearrange(x, 'b n d -> b (n d)')

        return x

    def translate_l(self, img):
        b, t, n = img.shape
        trans_x, small_trans_x = self.MT.get_latent_t(img)

        assert self.type == 'cat'
        trans_x = rearrange(trans_x, 'b n t d -> (b t) n d')
        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b * t)  # [(b t) n ed]

        x = trans_x + embed_x
        x = self.dropout(x)
        x, weights = self.S_transformer(x)  # [(b t) n ed+td]

        x = self.bottolneck(x)  # [(b t) n 1]

        return x  # b n d

    def forward(self, img):
        if self.ssl == True:
            return self.ssl_forward(img)
        else:
            return self.supervised_forward(img)


class Neural_ViT_Benchmark(nn.Module):
    """neuron amount depedent, mimic NDT arch"""
    def __init__(self, *,
                 num_classes,  # amount of final classification cls
                 depth,  # vit depth
                 heads,  # vit heads
                 neuron=160,
                 mlp_expend=2,  # vit mlp dim (typically 4 times expand in vision, here 2 is better)
                 dim_head=64,  # default is 64, better than 16/32/96
                 dropout=0.5,  # adjust from 0.2 to 0.5 -- performance varies
                 final_dim=None,
                 ssl=False,
                 ssl_ratio=1/6,
                 ):
        super().__init__()
        if final_dim == None:
            final_dim = neuron
        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes

        self.ssl = ssl

        mood = 'no'

        if mood != 'test':
            # neuron --> dim
            self.NDT_linear = nn.Linear(neuron, 2*neuron)
            self.NDT_transformer = Transformer(2*neuron,
                                               depth,
                                               heads,
                                               dim_head,
                                               2*neuron*mlp_expend,
                                               dropout,
                                               offset=False)
            self.NDT_temp_embed = nn.Parameter(torch.randn(1, 2, 2*neuron))

            self.NDT_back = nn.Linear(2*neuron, final_dim)
            self.NDT_cls = nn.Linear(final_dim, num_classes)
        else:
            print("NDT is in test mode")
            self.NDT_linear = nn.Linear(neuron, final_dim)
            self.NDT_transformer = Transformer(final_dim,
                                               depth, heads, 64, final_dim*mlp_expend, dropout,
                                               offset=False)
            self.NDT_temp_embed = nn.Parameter(torch.randn(1, 8, final_dim))

            self.NDT_back = nn.Identity()
            self.NDT_cls = nn.Linear(final_dim, num_classes)

        self.mask_token = nn.Parameter(torch.randn(2*neuron))
        self.masking_ratio = ssl_ratio
        self.final_activation = nn.Softplus()

    def latents(self, img):
        b, t, n = img.shape

        x = self.NDT_linear(img)  # b t dim
        temp_token = repeat(self.NDT_temp_embed, '() t d -> b t d', b=b)
        x = x + temp_token

        x, weights = self.NDT_transformer(x)  # [b t dim]
        x = self.NDT_back(x)

        return rearrange(x, 'b t l -> (b t) l')

    def super_forward(self, img, head=None):
        b, t, n = img.shape

        x = self.NDT_linear(img)  # b t dim
        temp_token = repeat(self.NDT_temp_embed, '() t d -> b t d', b=b)
        x = x + temp_token

        x, weights = self.NDT_transformer(x)  # [b t dim]
        x = self.NDT_back(x)
        x = self.NDT_cls(x)  # [b t cls]

        return rearrange(x, 'b t cls -> (b t) cls'), x

    def ssl_forward(self, img):

        b, t, n = img.shape
        device = img.device
        x = self.NDT_linear(img)  # b t dim

        # create masked_x and unmasked_x
        num_masked = int(self.masking_ratio * t)
        rand_indices = torch.rand(b, t, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(b, device=device)[:, None]
        unmasked_x = x[batch_range, unmasked_indices]
        masked_x = img[batch_range, masked_indices]

        embed_x = repeat(self.NDT_temp_embed, '() n ed -> b n ed', b=b)
        unmasked_embed_x = embed_x[batch_range, unmasked_indices]
        masked_embed_x = embed_x[batch_range, masked_indices]

        masked_tokens = repeat(self.mask_token, 'd -> b n d', b=b, n=num_masked)
        masked_tokens = masked_tokens + masked_embed_x
        unmasked_x = unmasked_x + unmasked_embed_x

        tokens = torch.cat((masked_tokens, unmasked_x), dim=1)
        x, weights = self.NDT_transformer(tokens)  # [b t dim]

        x = self.NDT_back(x)
        pred_masked_values = x[:, :num_masked]
        pred_masked_values = self.final_activation(pred_masked_values)

        recon_loss = self.reconstruction_loss(pred_masked_values, masked_x)
        return recon_loss, x

    def forward(self, img):
        if self.ssl:
            return self.ssl_forward(img)
        else:
            return self.super_forward(img)

    @staticmethod
    def reconstruction_loss(x, x_recon, distribution='poisson'):
        '''
            VAE works the best with bernoulli loss
            i-VAE works the best with poisson loss
        '''
        batch_size = x.size(0)  # [256 B, 163]
        assert batch_size != 0

        if distribution == 'bernoulli':  #
            recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
        elif distribution == 'weighted_bernoulli':
            weight = torch.tensor([0.1, 0.9]).to("cuda")  # just a label here
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
            # print(x_recon)
            x_recon = x_recon.clamp(min=1e-7, max=1e7)
            recon_loss = torch.sum(x_recon - x * torch.log(x_recon)).div(batch_size)
            # print(torch.sum(x_recon))
        else:
            recon_loss = None

        return recon_loss



