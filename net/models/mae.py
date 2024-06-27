# Copyright (c) Ren8394, Wei-Lun Chen
# Adjust MAE-ViT model to accept non-square input data and patch size.
# All rights reserved.
# -------------------------------------------
# Reference:
# Meta MAE: https://github.com/facebookresearch/mae/
# ViT: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# -------------------------------------------
from einops import rearrange, repeat
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block
import torch
import torch.nn as nn

from net.layers.pos_encode import PositionEncoding_2Dto1D
from net.layers.adapter import Residual_Adapter
from net.giants.swin_transformer_block import SwinTransformerBlock

class MAE(nn.Module):
    def __init__(
        self, in_shape, patch_size,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        use_mask_2d=False
    ):
        """
        Masked Autoencoder w/ Vision Transformer (MAE-ViT)
        
        Args:
            in_shape (tuple): input shape of the image (C, H, W)
            patch_size (tuple): size of the patch (H, W)
            embed_dim (int): ViT embedding dimension
            depth (int): ViT encoder depth
            num_heads (int): number of attention heads in ViT encoder
            decoder_embed_dim (int): decoder embedding dimension
            decoder_depth (int): ViT decoder depth
            decoder_num_heads (int): number of attention heads in decoder
            use_mask_2d (bool): use 2D mask or not. 
                If True, mask_row_ratio and mask_col_ratio are used in forward_encoder. 
                If False, mask_ratio w/ unstructure mask is used in forward_encoder.
        """
        super(MAE, self).__init__()
        self.in_channel, self.in_height, self.in_width = in_shape
        self.patch_height, self.patch_width = patch_size
        self.encoder_embed_dim, self.encoder_depth, self.encoder_num_heads = encoder_embed_dim, encoder_depth, encoder_num_heads
        self.decoder_embed_dim, self.decoder_depth, self.decoder_num_heads = decoder_embed_dim, decoder_depth, decoder_num_heads
        self.use_mask_2d = use_mask_2d

        self.grid_row, self.grid_col = self.in_height // self.patch_height, self.in_width // self.patch_width

        self.patch_embed = PatchEmbed(
            img_size=(self.in_height, self.in_width),
            patch_size=(self.patch_height, self.patch_width),
            in_chans=self.in_channel,
            embed_dim=encoder_embed_dim,
        )

        # encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.encoder_pos_embed = PositionEncoding_2Dto1D(encoder_embed_dim, (self.grid_row, self.grid_col), cls_token=True, requires_grad=False)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=encoder_embed_dim, num_heads=encoder_num_heads, mlp_ratio=4.0, qkv_bias=True, qk_norm=None)
            for _ in range(encoder_depth)])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = PositionEncoding_2Dto1D(decoder_embed_dim, (self.grid_row, self.grid_col), cls_token=True, requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=4.0, qkv_bias=True, qk_norm=None)
            for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, self.in_channel * self.patch_height * self.patch_width, bias=True)

        # initialize weights
        self.apply(self._init_weights)

    def patchify(self, x: torch.Tensor):
        """
        Patchify the input data tensor.

        Args:
            x (torch.Tensor): input data tensor (B, C, H, W)
        Returns:
            torch.Tensor: patchified data tensor (B, L, PH*PW*C)
        """
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=self.patch_height, pw=self.patch_width)
        return x

    def unpatchify(self, x: torch.Tensor):
        """
        Unpatchify the input data tensor.

        Args:
            x (torch.Tensor): input data tensor (B, L, PH*PW*C)
        Returns:
            torch.Tensor: unpatchified data tensor (B, C, H, W)
        """
        x = rearrange(x, "b (h w) (ph pw c) -> b c (h ph) (w pw)", h=self.in_height, w=self.in_width, ph=self.patch_height, pw=self.patch_width)
        return x

    def random_mask(self, x: torch.Tensor, mask_ratio: float):
        """
        Randomly mask the input data tensor.

        Args:
            x (torch.Tensor): input data tensor (B, L, D)
            mask_ratio (float): mask ratio
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # noise for mask
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # generate mask (0 - keep, 1 - mask)
        mask = torch.ones([B, L], device=x.device)
        mask[:, len_keep:] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # mask the input data
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore

    def random_mask_2d(self, x: torch.Tensor, mask_row_ratio: float, mask_col_ratio: float):
        """
        Randomly mask the input data tensor through row and column.

        Args:
            x (torch.Tensor): input data tensor (B, L, D)
            mask_row_ratio (float): mask ratio for row
            mask_col_ratio (float): mask ratio for column
        """
        B, L, D = x.shape
        num_row, num_col = self.grid_row, self.grid_col

        len_keep_row = int(num_row * (1 - mask_row_ratio))
        len_keep_col = int(num_col * (1 - mask_col_ratio))

        # noise for mask in row
        noise_row = torch.rand(B, num_row, device=x.device)
        ids_shuffle_row = torch.argsort(noise_row, dim=1)
        ids_restore_row = torch.argsort(ids_shuffle_row, dim=1)
        # noise for mask in column
        noise_col = torch.rand(B, num_col, device=x.device)
        ids_shuffle_col = torch.argsort(noise_col, dim=1)
        ids_restore_col = torch.argsort(ids_shuffle_col, dim=1)

        # generate mask (0 - keep, 1 - mask)
        mask_row = torch.ones([B, num_row], device=x.device)
        mask_row[:, len_keep_row:] = 0
        mask_row = torch.gather(mask_row, dim=1, index=ids_restore_row).unsqueeze(1).repeat(1, num_col, 1)  # (B, num_col, num_row)
        mask_col = torch.ones([B, num_col], device=x.device)
        mask_col[:, len_keep_col:] = 0
        mask_col = torch.gather(mask_col, dim=1, index=ids_restore_col).unsqueeze(2).repeat(1, num_row, 1).permute(0, 2, 1)  # (B, num_col, num_row)
        # combine row and column mask
        mask = 1 - (1 - mask_col) * (1 - mask_row)  # (B, num_col, num_row)

        # get ids to keep, and restore
        id2res = torch.Tensor(list(range(B * num_col * num_row))).reshape(B, num_col, num_row).to(x.device)
        id2res = id2res + 666 * mask
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep=id2res2.flatten(start_dim=1)[:, :len_keep_row * len_keep_col]
        ids_restore = torch.argsort(ids_keep.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        # mask the input data
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.1, mask_row_ratio: float = 0.1, mask_col_ratio: float = 0.1):
        B = x.shape[0]
        # patch embedding
        x = self.patch_embed(x)

        # positional encoding w/o cls token
        x = x + self.encoder_pos_embed.pe[:, 1:, :]
        
        # mask the input data
        if self.use_mask_2d:
            x_masked, mask, ids_restore = self.random_mask_2d(x, mask_row_ratio, mask_col_ratio)
        else:
            x_masked, mask, ids_restore = self.random_mask(x, mask_ratio)

        # add cls token
        cls_token = self.cls_token + self.encoder_pos_embed.pe[:, :1, :]
        cls_token = repeat(cls_token, "() n d -> b n d", b=B)

        x = torch.cat((cls_token, x_masked), dim=1)
        # encoder
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):        
        # token embedding
        x = self.decoder_embed(x)
        B, L, D = x.shape

        # add mask token
        mask_token = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - L, 1)
        x_no_cls = torch.cat((x[:, 1:, :], mask_token), dim=1)
        x_no_cls = torch.gather(x_no_cls, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x = torch.cat((x[:, :1, :], x_no_cls), dim=1)

        # position encoding
        x = x + self.decoder_pos_embed.pe

        # decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_head(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, x, mask_ratio: float = 0.1, mask_row_ratio: float = 0.1, mask_col_ratio: float = 0.1):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): input data tensor (B, C, H, W)
            mask_ratio (float): mask ratio, used when use_mask_2d is False
            mask_row_ratio (float): mask ratio for row, used when use_mask_2d is True
            mask_col_ratio (float): mask ratio for column, used when use_mask_2d is True

        """
        x_masked, mask, ids_restore = self.forward_encoder(x, mask_ratio, mask_row_ratio, mask_col_ratio)
        x_reconstructed = self.forward_decoder(x_masked, ids_restore)
        return x_reconstructed, mask, x_masked

    def forward_loss(self, x, y, mask):
        """
        Compute the loss function.

        Args:
            x (torch.Tensor): input data tensor (B, C, H, W)
            y (torch.Tensor): target data tensor (B, L, ph*pw*C)
            mask (torch.Tensor): mask tensor (B, L), 0 is kept, 1 is masked
        """
        x_patch = self.patchify(x)
        
        loss = (y - x_patch) ** 2
        loss = loss.mean(dim=-1)                   # (B, L), mean loss for each patch
        loss = (loss * mask).sum() / mask.sum()    # mean loss for each masked patch
        
        return loss

    def _init_weights(self, layer):
        # initialize patch embedding as nn.Linear
        patch_weight = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(patch_weight.view(patch_weight.size(0), -1))

        # initialize token
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0)
            nn.init.constant_(layer.weight, 1.0)

class MAE_Adapter(MAE):
    def __init__(
        self, 
        use_adapter=True, adapter_size=128, 
        **kwargs,
    ):
        """
        Masked Autoencoder w/ Vision Transformer (MAE-ViT)
        Add residual adapter to the encoder.
        
        Args:
            use_adapter (bool): use adapter or not
            adapter_size (int): adapter size
            **kwargs: arguments for MAE_ViT
        """
        super(MAE_Adapter, self).__init__(**kwargs)

        # add residual adapter
        self.use_adapter = use_adapter

        self.adapter = Residual_Adapter(self.encoder_embed_dim, adapter_size)
        if use_adapter:
            self.adapter.requires_grad_(True)
        else:
            self.adapter.requires_grad_(False)

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.1, mask_row_ratio: float = 0.1, mask_col_ratio: float = 0.1):
        B = x.shape[0]
        # patch embedding
        x = self.patch_embed(x)

        # positional encoding w/o cls token
        x = x + self.encoder_pos_embed.pe[:, 1:, :]
        
        # mask the input data
        if self.use_mask_2d:
            x_masked, mask, ids_restore = self.random_mask_2d(x, mask_row_ratio, mask_col_ratio)
        else:
            x_masked, mask, ids_restore = self.random_mask(x, mask_ratio)

        # add cls token
        cls_token = self.cls_token + self.encoder_pos_embed.pe[:, 0, :]
        cls_token = repeat(cls_token, "() n d -> b n d", b=B)

        x = torch.cat((cls_token, x_masked), dim=1)
        # encoder / adapter
        for i, block in enumerate(self.encoder_blocks):
            if i == len(self.encoder_blocks) - 1:
                x = block(x)
                x = self.adapter(x)
            else:
                x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

class MAE_Swin(MAE):
    def __init__(
        self,
        **kwargs,           
    ):
        super(MAE_Swin, self).__init__(**kwargs)

        window_size = (4, 4)
        feat_size = (self.grid_row, self.grid_col)

        # swin transformer decoder
        decoder_module = []
        for index in range(16):
            if (index % 2) == 0:
                shift_size = (0, 0)
            else:
                shift_size = (2, 0)
            decoder_module.append(
                SwinTransformerBlock(
                    dim=self.decoder_embed_dim,
                    num_heads=16,
                    feat_size=feat_size,
                    window_size=window_size,
                    shift_size=shift_size,
                    norm_layer=nn.LayerNorm,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_module)

    def forward_decoder(self, x, ids_restore):        
        # token embedding
        x = self.decoder_embed(x)
        B, L, D = x.shape

        # add mask token
        mask_token = self.mask_token.repeat(B, ids_restore.shape[1] + 1 - L, 1)
        x_no_cls = torch.cat((x[:, 1:, :], mask_token), dim=1)
        x_no_cls = torch.gather(x_no_cls, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x = torch.cat((x[:, :1, :], x_no_cls), dim=1)

        # position encoding
        x = x + self.decoder_pos_embed.pe

        # remove cls token
        x = x[:, 1:, :]

        # decoder
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_head(x)

        return x