from einops.layers.torch import Rearrange
from my_transformers.model_utils import *


class Subset_transformer_add(nn.Module):
    def __init__(self,
                 out_dim=160,
                 num_patches=8,
                 num_neurons=20,
                 trans_dim=8,
                 embed_dim=8,
                 mlp_dim=8, depth=1, heads=4):
        """
        out_dim -- is the dimension that needs to be out
        num_patches -- number of patches
        num_neurons -- number of neurons within a patch
        trans_dim -- linear dim for each neuron
        embed_dim -- embedding dim for each neuron
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.trans_dim, self.embed_dim = trans_dim, embed_dim
        self.out_dim = out_dim

        # linear layer
        self.neuron_linear = nn.Linear(1, trans_dim)
        self.neuron_id = nn.Parameter(torch.randn(1, num_patches, num_neurons, embed_dim))

        self.transformer = Transformer(dim=trans_dim+embed_dim,
                                       depth=depth,
                                       heads=heads,
                                       dim_head=trans_dim+embed_dim,
                                       mlp_dim=mlp_dim)

        self.to_trans_input = Rearrange('b n pn vn -> (b n) pn vn')
        self.to_trans_output = Rearrange('(b n) pn vn -> b n pn vn', n=8)

        if out_dim == int((trans_dim + embed_dim) * num_neurons):
            self.to_token = nn.Sequential(Rearrange('b n pn vn -> b n (pn vn)'))
        else:
            self.to_token = nn.Sequential(Rearrange('b n pn vn -> b n (pn vn)'),
                                          nn.Linear(int((trans_dim + embed_dim) * num_neurons), out_dim))

    def forward(self, patches):
        """
        a transformer version of patch to tokens
        patches: (b n pn), pn=patch_height, n is the total amount of patches
        """
        patches = patches[:, :, :, None]
        patches = self.neuron_linear(patches)  # (b n pn trans_dim)

        batch = patches.shape[0]
        feature_id_rp = repeat(self.neuron_id, '1 n pn d -> b n pn d', b=batch)
        patches = torch.cat([patches, feature_id_rp], dim=-1)

        patches = self.to_trans_input(patches)
        patches, weights = self.transformer(patches)
        patches = self.to_trans_output(patches)
        tokens = self.to_token(patches)

        return tokens, weights
