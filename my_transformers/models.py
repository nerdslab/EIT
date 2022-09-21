import torch.nn.functional as F

# from vit_pytorch import ViT, MAE
from vit_pytorch.vit import Transformer

from einops.layers.torch import Rearrange

from my_transformers.model_utils import *

"""a copy of ViT and MAE to understand their implementation"""
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_height, self.patch_width = patch_height, patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]  # separated
        pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.A_style = nn.Parameter(torch.randn(1, num_patches - 1, decoder_dim))  # should be [1*batch, 1*num_patches, dim]
        self.B_style = nn.Parameter(torch.randn(1, num_patches - 1, decoder_dim))

        self.decoder = Transformer(dim=decoder_dim, depth=decoder_depth, heads=decoder_heads, dim_head=decoder_dim_head, mlp_dim=decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)  # 512 --> 32*32*3

        self.img_shuffle_perm = torch.randperm(64*64)
        self.pool = 'mean'

    def forward(self, img, style=False):
        device = img.device

        Shuffle = True
        if Shuffle:
            #img = img_shuffule(img, perm=self.img_shuffle_perm)
            img = img_grid(img, grid=8, patch=8)

        # get patches
        patches = self.to_patch(img)  # b 8*8 32*32*3
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions
        tokens = self.patch_to_emb(patches)  # b 8*8 1024
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]  # without the cls token

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]  # torch.Size([8, 16, 1024])

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]  # original images [8, 48, 32*32*3]

        # attend with vision transformer
        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)  # batch, numb_masked 48, 512
        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)  # batch, 64, 512

        if style is not False:
            if style == "A":
                add_style = self.A_style
            elif style == "B":
                add_style = self.B_style
            # print(decoder_tokens.shape, self.style.shape)
            decoder_tokens += add_style[:, :]

        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, -num_masked:]  # the first fews are removed
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        # recon img using the below func
        unshuffle = self._reconstrcuct_img_show(img.shape, rand_indices, num_masked, pred_pixel_values, patches,
                                                p_height=self.encoder.patch_height, p_width=self.encoder.patch_width)

        if Shuffle:
            #unshuffle = img_unshuffle(unshuffle, perm=self.img_shuffle_perm)
            unshuffle = img_ungrid(unshuffle, grid=8, patch=8)

        #return recon_loss, pred_pixel_values, unshuffle
        encoded_tokens = encoded_tokens.mean(dim=1) if self.pool == 'mean' else encoded_tokens[:, 0]
        return recon_loss, encoded_tokens, unshuffle


    def clean_recon(self, img):
        ...

    @staticmethod
    def _reconstrcuct_img_show(img_shape, rand_indices, num_masked, pred_pixel_values, patches,
                               p_height=32, p_width=32, original_unmask=True):
        """
        img_shape as the original img.shape
        rand_indices and num_masked
        pred_values as the img to be used
        patches as patch info  # b 8*8 32*32*3
        """
        batch, num_patches, *_ = patches.shape
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        batch_range = torch.arange(batch, device=pred_pixel_values.device)[:, None]
        unmasked_patches = patches[batch_range, unmasked_indices]

        if original_unmask:
            recon_data = torch.cat([pred_pixel_values, unmasked_patches], dim=1)
        else:
            unmasked_zeros = torch.zeros(size=unmasked_patches.shape, device=unmasked_patches.device)
            recon_data = torch.cat([pred_pixel_values, unmasked_zeros], dim=1)

        un_shuffle = torch.zeros(recon_data.shape).to(patches.device)
        un_shuffle[batch_range, rand_indices] = recon_data

        to_img = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                           c=img_shape[1], p1=p_height, p2=p_width, h=int(img_shape[-1]/p_width))
        un_shuffle = to_img(un_shuffle)
        return un_shuffle

