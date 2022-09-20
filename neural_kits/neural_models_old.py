import torch.nn.functional as F
from einops.layers.torch import Rearrange

from my_transformers.model_utils import *


"""the benchmark (bm) model is a neuron-level ViT"""
"""the reference model (ours) is a eiffel ViT"""
class Neural_ViT_bm(nn.Module):
    def __init__(self, *,
                 neuron,  # amount of neurons
                 num_classes,  # amount of final classification cls
                 trans_dim,  # transformation dim of single neuron
                 embed_dim,  # embedding dim of single neuron
                 depth,  # vit depth
                 heads,  # vit heads
                 mlp_expend=1,  # vit mlp dim (typically 4 times expand)
                 pool='mean',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 trans_embed_relation='cat',
                 ):
        super().__init__()
        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert trans_embed_relation in {'add', 'cat'}
        assert trans_embed_relation in {'cat'}, 'add embed not implemented yet'

        self.to_patch = Rearrange('b (n pn) -> b n pn', pn=1)
        self.patch_to_token = nn.Linear(1, trans_dim)
        # In the benchmark model, pos embedding works the same as the neuron embedding
        self.neuron_embed = nn.Parameter(torch.randn(1, neuron, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim+trans_dim))
        self.dim = embed_dim+trans_dim

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, depth, heads, dim_head, mlp_expend*self.dim, dropout)
        self.depth, self.heads, self.dim_head, self.mlp_expend = depth, heads, dim_head, mlp_expend

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim+trans_dim),
            nn.Linear(embed_dim+trans_dim, num_classes)
        )

    def forward(self, img):
        """img size is [batch_size, neuron_amount]"""
        device = img.device
        b, n = img.shape

        x = self.to_patch(img)  # [b n 1]
        trans_x = self.patch_to_token(x)  # [b n td]
        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b)  # [b n ed]
        cls_token = repeat(self.cls_token, '() 1 d -> b 1 d', b=b)

        x = torch.cat([trans_x, embed_x], dim=-1)
        x = torch.cat([cls_token, x], dim=-2) # [b n+1 ed+td]
        x = self.dropout(x)

        x, weights = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x), {"weights":weights}


class Neural_ViT_bm_ltemp(nn.Module):
    def __init__(self, *,
                 neuron,  # amount of neurons
                 num_classes,  # amount of final classification cls
                 single_dim, # single neuron dim
                 embed_dim,  # dim of all neurons times total_time
                 depth,  # vit depth
                 heads,  # vit heads
                 mlp_expend=1,  # vit mlp dim (typically 4 times expand)
                 pool='mean',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 ):
        super().__init__()
        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes
        t = 8
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert t*single_dim == embed_dim

        self.to_temp = Rearrange('b n t -> (b n) t')
        self.single_embed = nn.Linear(1, single_dim)
        self.T_transformer = Transformer(single_dim, 2, heads, dim_head, mlp_expend * single_dim, dropout)

        # In the benchmark model, pos embedding works the same as the neuron embedding
        self.neuron_embed = nn.Parameter(torch.randn(1, neuron, embed_dim))  # based on total amount of neuron
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.S_transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_expend*embed_dim, dropout)

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.mlp_head_train = nn.Sequential(nn.Linear(single_dim, 2))  # 8*16
        self.mlp_head_try = nn.Sequential(
            # nn.LayerNorm(neuron*single_dim),
            nn.Linear(neuron*2, num_classes)
        )
        self.forward_type = 't'

    def forward(self, img):
        """img size is [batch_size, time, neuron_amount]"""
        device = img.device
        b, t, n = img.shape

        # feed into t transformer, without temperol id
        img = torch.transpose(img, -1, -2)  # [batch, neuron, time]
        x = self.to_temp(img)[:, :, None] # [(batch, neuron), time]
        trans_x = self.single_embed(x) # [(batch, neuron), time, dim]
        trans_x, t_weights = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]

        # transformmmmmmm
        trans_x = rearrange(trans_x, '(b n) t d -> b n (t d)', b=b)  # [batch, neuron, (time, dim)]

        # embed with neuron id and feed it into spatial transformer
        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b)  # [b n ed]
        cls_token = repeat(self.cls_token, '() 1 d -> b 1 d', b=b)
        x = trans_x + embed_x
        x = torch.cat([cls_token, x], dim=-2) # [b n+1 ed+td]
        x = self.dropout(x)
        x, weights = self.S_transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        if self.forward_type == 't':
            trans_x = rearrange(trans_x, 'b n (t d) -> b n t d', t=t)
            trans_x = self.mlp_head_train(trans_x)
            x = rearrange(trans_x, 'b n t d -> b t (n d)')
            return rearrange(self.mlp_head_try(x), 'b t cls -> (b t) cls'), x
        elif self.forward_type == 's':
            return self.mlp_head(x), {"weights": weights}

    def get_latent_t(self, img):
        device = img.device
        b, t, n = img.shape

        # feed into t transformer, without temperol id
        img = torch.transpose(img, -1, -2)  # [batch, neuron, time]
        x = self.to_temp(img)[:, :, None]  # [(batch, neuron), time]
        trans_x = self.single_embed(x)  # [(batch, neuron), time, dim]
        trans_x, t_weights = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]

        trans_x = rearrange(trans_x, '(b n) t d -> b n (t d)', b=b)  # [batch, neuron, (time, dim)]

        trans_x = rearrange(trans_x, 'b n (t d) -> b n t d', t=t)
        small_trans_x = self.mlp_head_train(trans_x)
        return trans_x, small_trans_x




class Neural_ViT_bm_multireso(nn.Module):
    """multi resolution Neural ViT"""
    def __init__(self, *,
                 neuron,  # amount of neurons
                 num_classes,  # amount of final classification cls
                 trans_dim,  # transformation dim of single neuron
                 embed_dim,  # embedding dim of single neuron
                 depth,  # vit depth
                 heads,  # vit heads
                 mlp_expend=1,  # vit mlp dim (typically 4 times expand)
                 pool='mean',
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 ):
        super().__init__()
        self.image_height = neuron  # 160, 16
        self.num_classes = num_classes
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling'

        self.dim = embed_dim

        self.to_patch = Rearrange('b n t -> (b n) t')
        self.patch_to_token = nn.Linear(1, 16)
        self.T_transformer = Transformer(16, 2, 16, dim_head, mlp_expend * 16, dropout)

        # In the benchmark model, pos embedding works the same as the neuron embedding
        self.neuron_embed = nn.Parameter(torch.randn(1, neuron, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.posi_embed = nn.Parameter(torch.randn(1, neuron+1, embed_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, depth, heads, dim_head, mlp_expend*self.dim, dropout)
        self.depth, self.heads, self.dim_head, self.mlp_expend = depth, heads, dim_head, mlp_expend

        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.mlp_head_try = nn.Sequential(
            nn.LayerNorm(160*16),
            nn.Linear(160*16, num_classes)
        )

    def forward(self, img):
        """img size is [batch_size, time, neuron_amount]"""
        device = img.device
        b, t, n = img.shape

        img = torch.transpose(img, -1, -2)  # [batch, neuron, time]
        x = self.to_patch(img)[:, :, None] # [(batch, neuron), time]
        # print(x.shape)
        trans_x = self.patch_to_token(x) # [(batch, neuron), time, dim]

        trans_x, _ = self.T_transformer(trans_x)  # [(batch, neuron), time, dim]
        trans_x = rearrange(trans_x, '(b n) t d -> b n (t d)', b=b)  # [batch, neuron, (time, dim)]

        embed_x = repeat(self.neuron_embed, '() n ed -> b n ed', b=b)  # [b n ed]
        cls_token = repeat(self.cls_token, '() 1 d -> b 1 d', b=b)

        x = trans_x + embed_x
        x = torch.cat([cls_token, x], dim=-2) # [b n+1 ed+td]
        # x += self.posi_embed[:, :(n + 1)]

        x = self.dropout(x)
        x, weights = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        #x = rearrange(trans_x, 'b n (t d) -> b t (n d)', t=t)
        #return rearrange(self.mlp_head_try(x), 'b t dim -> (b t) dim'), {"weights": weights}

        return self.mlp_head(x), {"weights":weights}


class Neural_ViT_eiffel(Neural_ViT_bm):
    def __init__(self, lower_depth=1, patch=20, higher_dim=160,
                 masking_ratio=0.5,
                 training_type='super', **kwargs):
        super().__init__(**kwargs)
        self.dim = 8
        self.higher_dim = higher_dim

        self.masking_ratio = masking_ratio
        self.mask_neuron = nn.Parameter(torch.randn(1, 160, self.dim)) # 160 neuron, each has self.dim

        self.patch_higher = patch  # e.g. 20
        self.num_patches_higher = self.image_height // patch  # e.g. 8

        self.sparcify = Rearrange('b (pn p) d -> (b pn) p d', pn=self.num_patches_higher)
        self.un_sparcify = Rearrange('(b pn) p d -> b (pn p) d', pn=self.num_patches_higher)

        self.to_higher_patch = Rearrange('b (n pn) d -> b n (pn d)', pn=patch)
        # self.embed_to_higher_patch = nn.Linear(patch*self.dim, higher_dim)
        # I just defined smth without using it and it still impact my model performance ??

        # assert patch*self.dim == higher_dim
        self.embed_to_higher_patch = nn.Identity() if patch*self.dim == higher_dim else nn.Linear(patch * self.dim, higher_dim)

        # self.neuron_embed = nn.Parameter(torch.randn(1, neuron, embed_dim))
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, higher_dim))
        self.pos_embed_trans = nn.Identity() if patch*self.dim == higher_dim else nn.Linear(patch * self.dim, higher_dim)
        # self.pos_embedding = self.neuron_embed.clone()
        self.higher_cls_token = nn.Parameter(torch.randn(1, 1, higher_dim))

        self.lower_transformer = Transformer(self.dim, lower_depth, self.heads, 16, self.mlp_expend*self.dim)
        self.higher_transformer = Transformer(higher_dim, int(self.depth-lower_depth), self.heads, self.dim_head, self.mlp_expend*higher_dim)

        self.decoder = Transformer(dim=160, depth=1, heads=4, dim_head=self.dim_head, mlp_dim=80*4)
        self.to_pixels = nn.Linear(self.dim, 1)  # 512 --> 32*32*3
        self.final_activation = nn.Softplus()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(higher_dim),
            nn.Linear(higher_dim, self.num_classes)
        )

        self.type = training_type

    def forward_super(self, img):
        """img size is [batch_size, neuron_amount]"""
        device = img.device
        b, neuron_amount = img.shape

        x = self.to_patch(img)  # [b n 1]
        trans_x = self.patch_to_token(x)  # [b n td]
        x = trans_x + self.neuron_embed[:, :]

        x = self.to_higher_patch(x)
        x = self.embed_to_higher_patch(x)
        cls_token = repeat(self.higher_cls_token, '() 1 d -> b 1 d', b=b)
        x = torch.cat([cls_token, x], dim=-2)
        _, n, _ = x.shape

        pos_embedding = rearrange(self.neuron_embed.clone(), '1 (pn n) d -> 1 pn (n d)', n=20)
        pos_embedding = self.pos_embed_trans(pos_embedding)
        pos_embedding = torch.cat([self.cls_embedding, pos_embedding], dim=-2)
        x += pos_embedding[:, :(n + 1)]

        x, weights = self.higher_transformer(x)  # b pn (n d)
        return x, weights

    def forward_unsuper(self, img):
        """img size is [batch_size, neuron_amount]"""
        device = img.device
        b, neuron_amount = img.shape

        num_masked = int(self.masking_ratio * neuron_amount)
        rand_indices = torch.rand(b, neuron_amount, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        x = self.to_patch(img)  # [b n 1]
        trans_x = self.patch_to_token(x)  # [b n td]

        # masking procedures
        batch_range = torch.arange(b, device=device)[:, None]
        mask_x = repeat(self.mask_neuron, '() n d -> b n d', b=b)
        masked_real_x = x[batch_range, masked_indices]
        new_x = torch.cat([mask_x[batch_range, masked_indices], trans_x[batch_range, unmasked_indices]], dim=-2)

        empty_x = torch.zeros(trans_x.shape, device=device)
        empty_x[batch_range, rand_indices] = new_x.clone()
        trans_x = empty_x.clone()

        x = trans_x + self.neuron_embed[:, :]

        x = self.to_higher_patch(x)
        # x = self.embed_to_higher_patch(x)
        cls_token = repeat(self.higher_cls_token, '() 1 d -> b 1 d', b=b)
        x = torch.cat([cls_token, x], dim=-2)
        _, n, _ = x.shape

        pos_embedding = rearrange(self.neuron_embed.clone(), '1 (pn n) d -> 1 pn (n d)', n=20)
        pos_embedding = torch.cat([self.cls_embedding, pos_embedding], dim=-2)
        x += pos_embedding[:, :(n + 1)]

        x, weights = self.higher_transformer(x)  # b pn (n d)

        decoded_x, _ = self.decoder(x)  # b pn (n d)
        decoded_x = rearrange(decoded_x, 'b p (n d) -> b (p n) d', n=20)
        mask_tokens = decoded_x[batch_range, masked_indices]  # the first fews are removed

        pred_pixel_values = self.to_pixels(mask_tokens)
        pred_pixel_values = self.final_activation(pred_pixel_values)

        encoded_x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return (pred_pixel_values, masked_real_x), encoded_x, _

    def forward(self, img):
        """img size is [batch_size, neuron_amount]"""
        if self.type == 'super':
            x, weights = self.forward_super(img)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            individual_weights = None
            return self.mlp_head(x), {"weights": weights, "individual_weights": individual_weights}
        elif self.type == 'unsuper':
            return self.forward_unsuper(img)
        else:
            raise NotImplementedError('wwwwwhat do you want??')


class Neural_ViT_with_temp(Neural_ViT_eiffel):
    def __init__(self, t=8, NDT_bm=False, statue='train', **kwargs):
        super().__init__(**kwargs)
        temp_size=t
        self.NDT_bm = NDT_bm

        self.to_spatial = Rearrange('b t n -> (b t) n')
        self.to_temperal = Rearrange('(b t) pn hd -> b t (pn hd)', t=temp_size)  # pn here is pn+1
        self.temp_transformer = Transformer(dim=(self.num_patches_higher + 1)*self.higher_dim,
                                            depth=1, heads=4, dim_head=320, mlp_dim=1280)
        self.to_original_spatial = Rearrange('b t (pn hd) -> (b t) pn hd', hd=self.higher_dim)

        self.NDT_linear = nn.Linear(160, 320)
        self.NDT_transformer = Transformer(dim=320, depth=4, heads=4, dim_head=320, mlp_dim=1280)
        self.NDT_mlp_head = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, self.num_classes)
        )

        self.statue = statue

    def forward_eiffel(self, img):
        """img size is [batch_size, time, neuron_amount] in this case"""
        b, t, n = img.shape

        img = self.to_spatial(img)  # (b t) n
        x, weights = self.forward_super(img)  # x shape [(b t) (pn+1) (n d)]
        x = self.to_temperal(x)  # [b t (pn+1)nd]

        x, temp_weights = self.temp_transformer(x)
        x = self.to_original_spatial(x)  # [(b t) (pn+1) hd]

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        # mlp head have a batch size [b*t], compute it with [b*t labels]
        return self.mlp_head(x), {"weights": weights, "temp_weights": temp_weights}

    def forward_NDT(self, img):
        b, t, n = img.shape

        x = self.NDT_linear(img) # b t dim
        x, weights = self.NDT_transformer(x)  # [b t dim]
        x = self.to_spatial(x)  # [(b t) dim]

        return self.NDT_mlp_head(x), {"weights": weights}

    def forward(self, img):
        if self.statue == 'train':
            assert len(img.shape) == 3
            if self.NDT_bm:
                return self.forward_NDT(img)
            else:
                return self.forward_eiffel(img)
        elif self.statue == 'test':
            # print("?")
            assert len(img.shape) == 2
            x, weights = self.forward_super(img)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            individual_weights = None
            return self.mlp_head(x), {"weights": weights, "individual_weights": individual_weights}


class Neural_ViT_couple_temp(nn.Module):
     def __init__(self, v, t=8):
         super().__init__()
         self.v = v

         self.to_spatial = Rearrange('b t n -> (b t) n')
         self.to_temperal = Rearrange('(b t) pn hd -> b t (pn hd)', t=temp_size)  # pn here is pn+1
         self.temp_transformer = Transformer(dim=(self.num_patches_higher + 1) * self.higher_dim,
                                             depth=1, heads=4, dim_head=320, mlp_dim=1280)
         self.to_original_spatial = Rearrange('b t (pn hd) -> (b t) pn hd', hd=self.higher_dim)

