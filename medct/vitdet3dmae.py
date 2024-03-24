# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_vitdet3dmae.ipynb.

# %% auto 0
__all__ = ['ViTDet3DMAEConfig', 'get_3d_position_embeddings', 'embed_spacings_in_position_embeddings',
           'ViTDet3DMAEPatchEmbeddings', 'ViTDet3DMAEEmbeddings', 'ViTDet3DMAEEncoder', 'ViTDet3DMAEModelOutput',
           'ViTDet3DMAEPreTrainedModel', 'ViTDet3DMAEModel', 'ViTDet3DMAEDecoderOutput', 'ViTDet3DMAEDecoder',
           'ViTDet3DMAEForPreTrainingOutput', 'ViTDet3DMAEForPreTraining']

# %% ../nbs/03_vitdet3dmae.ipynb 4
import torch
import numpy as np
import collections

from copy import deepcopy
from dataclasses import dataclass
from einops import rearrange, repeat
from .vitdet3d import VitDetConfig, VitDet3dEncoder, VitDet3dPreTrainedModel
from torch import nn
from transformers.models.vit_mae.modeling_vit_mae import ViTMAELayer
from transformers.utils import ModelOutput

# %% ../nbs/03_vitdet3dmae.ipynb 7
# Config inherits hyperparameters from both ViTDet and ViTMAE
class ViTDet3DMAEConfig(VitDetConfig):
    def __init__(
        self,
        attention_probs_dropout_prob=0.0,
        decoder_hidden_size=384,
        decoder_intermediate_size=1536,
        decoder_learnable_position_embeddings=False,
        decoder_num_attention_heads=12,
        decoder_num_hidden_layers=1,
        embed_spacing=False,
        hidden_dropout_prob=0.0,
        hidden_size=768,
        image_size=(32, 256, 256),
        intermediate_size=3072,
        learnable_position_embeddings=False,
        mask_ratio=0.0,
        norm_pix_loss=False,
        num_attention_heads=12,
        num_hidden_layers=12,
        patch_size=(4, 32, 32),
        pretraining_image_size=(32, 256, 256),
        pretraining=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_type = "vitdet3dmae"

        assert self.use_relative_position_embeddings is False, "use_relative_position_embeddings is not supported"
        self.use_relative_position_embeddings = False
        self.use_absolute_position_embeddings = True

        assert pretraining or mask_ratio == 0, "mask_ratio should be 0 if not pretraining"

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_intermediate_size = decoder_intermediate_size
        self.decoder_learnable_position_embeddings = decoder_learnable_position_embeddings
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.embed_spacing = embed_spacing
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.intermediate_size = intermediate_size
        self.learnable_position_embeddings = learnable_position_embeddings
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.patch_size = patch_size
        self.pretraining = pretraining
        self.pretraining_image_size = pretraining_image_size

        self.pretraining_grid_size, self.pretraining_num_patches = self.get_grid_size_and_num_patches(
            self.pretraining_image_size
        )

        self.perform_checks()

    def perform_checks(self):
        if not isinstance(self.patch_size, collections.abc.Iterable) or len(self.patch_size) != 3:
            raise ValueError("patch_size must be given as 3D iterable")

        assert (
            torch.remainder(torch.tensor(self.pretraining_image_size), torch.tensor(self.patch_size)).sum() == 0
        ), "Image size must be divisible by patch size"

        assert self.learnable_position_embeddings is False, "Learnable position embeddings are not supported yet"
        assert (
            self.decoder_learnable_position_embeddings is False
        ), "Learnable position embeddings are not supported yet"

        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

    def get_grid_size_and_num_patches(self, image_size):
        grid_size = (
            image_size[0] // self.patch_size[0],
            image_size[1] // self.patch_size[1],
            image_size[2] // self.patch_size[2],
        )
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return grid_size, num_patches

# %% ../nbs/03_vitdet3dmae.ipynb 12
def get_3d_position_embeddings(embedding_size, grid_size, patch_size=(1, 1, 1)):
    if embedding_size % 6 != 0:
        raise ValueError("embed_dim must be divisible by 6")

    d, h, w = grid_size

    grid_d = np.arange(d, dtype=np.float32)
    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h, grid_d)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, d, h, w])

    omega = np.arange(embedding_size // 6, dtype=float)
    omega /= embedding_size / 6.0
    omega = 1.0 / 10000**omega

    patch_multiplier = np.array(patch_size) / min(patch_size)

    position_embeddings = []
    for i, grid_subset in enumerate(grid):
        grid_subset = grid_subset.reshape(-1)
        out = np.einsum("m,d->md", grid_subset, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1) * patch_multiplier[i]
        position_embeddings.append(emb)

    position_embeddings = np.concatenate(position_embeddings, axis=1)
    position_embeddings = position_embeddings.reshape([1, embedding_size, d, h, w])
    position_embeddings = torch.from_numpy(position_embeddings).float()

    return position_embeddings

# %% ../nbs/03_vitdet3dmae.ipynb 13
def embed_spacings_in_position_embeddings(embeddings: torch.Tensor, spacings: torch.Tensor):
    assert spacings.ndim == 2, "Please provide spacing information for each batch element"
    _, embedding_size, _, _, _ = embeddings.shape
    embeddings = embeddings.clone() * repeat(spacings, f"B S -> B (S {int(embedding_size / 3)}) 1 1 1", S=3)

    return embeddings

# %% ../nbs/03_vitdet3dmae.ipynb 15
class ViTDet3DMAEPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTDet3DMAEConfig) -> None:
        super().__init__()

        self.config = config
        self.projection = nn.Conv3d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, pixel_values: torch.Tensor):
        num_channels = pixel_values.shape[1]
        if num_channels != self.config.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.config.num_channels} but got {num_channels}."
            )

        image_size = pixel_values.shape[2:]
        if len(image_size) != 3:
            raise ValueError("Image size must be 3D")

        if self.config.pretraining and self.training:
            if self.config.pretraining_image_size != image_size:
                raise ValueError(
                    "Make sure that the spatial dimensions of the pixel values match with the ones set in the "
                    f"configuration. Expected {self.config.image_size} but got {image_size}."
                )

        # (b, c, z, y, x)
        embeddings = self.projection(pixel_values)
        # (b, hidden_size, z, y, x)

        return embeddings

# %% ../nbs/03_vitdet3dmae.ipynb 18
class ViTDet3DMAEEmbeddings(nn.Module):
    def __init__(self, config: ViTDet3DMAEConfig):
        super().__init__()

        self.config = deepcopy(config)

        if self.config.pretraining:
            self.config.image_size = config.pretraining_image_size
        self.patch_embeddings = ViTDet3DMAEPatchEmbeddings(config)

        # Positional embeddings
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def uniform_sampling(self, embeddings: torch.Tensor):
        # embeddings: (b, hidden_size, z, y, x)
        B, _, Z, Y, X = embeddings.shape
        assert Z % 2 == 0 and Y % 2 == 0 and X % 2 == 0, f"Z, Y, X must be even. Got shape {embeddings.shape}"

        # Only works if mask_ratio is 0, 0.75, or 0.875
        assert self.config.mask_ratio in {0, 0.75, 0.875}, "Mask ratio must be 0.75 or 0.875 (or 0)"

        if self.config.mask_ratio == 0:
            masks = torch.ones(B, Z, Y, X, device=embeddings.device, dtype=torch.int8)
        else:
            masks = []
            for _ in range(len(embeddings)):
                # Perform uniform sampling. Select one patch in a 2x2x2 cube.
                # Also add the diagonally opposite patch if mask_ratio == 0.75.
                num_decisions = X * Y * Z // 8
                if self.config.mask_ratio == 0.75:
                    mask1 = np.eye(4, dtype="int").repeat(num_decisions, axis=0)  # (num_decisions * 4, 4)
                    np.random.shuffle(mask1)
                    mask1 = mask1[:num_decisions]  # (num_decisions, 4)
                    mask2 = mask1.copy()

                    mask2[:, (0, 3)] = mask2[:, (3, 0)]
                    mask2[:, (1, 2)] = mask2[:, (2, 1)]
                    mask = np.concatenate([mask1, mask2], axis=1)  # (num_decisions, 8)
                else:
                    mask = np.eye(8, dtype="int").repeat(num_decisions, axis=0)  # (num_decisions * 8, 8)
                    np.random.shuffle(mask)
                    mask = mask[:num_decisions]  # (num_decisions, 8)

                mask = rearrange(
                    mask, "(d h w) (p1 p2 p3) -> (d p1) (h p2) (w p3)", d=Z // 2, h=Y // 2, w=X // 2, p1=2, p2=2, p3=2
                )
                masks.append(mask)
            masks = np.stack(masks, axis=0)

            masks = torch.tensor(masks, device=embeddings.device, dtype=torch.int8)

        return masks

    def forward(self, pixel_values: torch.Tensor, spacings: torch.Tensor):
        # Get patch embeddings
        embeddings = self.patch_embeddings(pixel_values)
        # (b, hidden_size, z, y, x)

        # Get positional embeddings
        image_size = pixel_values.shape[2:]
        grid_size, _ = self.config.get_grid_size_and_num_patches(image_size)
        position_embeddings = get_3d_position_embeddings(self.config.hidden_size, grid_size)
        position_embeddings = position_embeddings.to(pixel_values.device)
        # (1, hidden_size, z, y, x)
        if self.config.embed_spacing:
            position_embeddings = embed_spacings_in_position_embeddings(position_embeddings, spacings)
            # (b, hidden_size, z, y, x)

        # Add positional embeddings
        B, C, Z, Y, X = embeddings.shape
        embeddings = embeddings + position_embeddings

        # Uniform sampling
        masks = self.uniform_sampling(embeddings)

        # Get masked embeddings
        new_Z, new_Y, new_X = Z, Y, X
        if self.config.mask_ratio == 0.875:
            new_Z, new_Y, new_X = Z // 2, Y // 2, X // 2
        elif self.config.mask_ratio == 0.75:
            new_Z, new_Y, new_X = Z, Y // 2, X // 2
        embeddings = torch.masked_select(embeddings, masks.unsqueeze(1).bool()).reshape(B, C, new_Z, new_Y, new_X)

        return embeddings, masks

# %% ../nbs/03_vitdet3dmae.ipynb 29
class ViTDet3DMAEEncoder(VitDet3dEncoder):
    pass

# %% ../nbs/03_vitdet3dmae.ipynb 30
@dataclass
class ViTDet3DMAEModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    masks: torch.LongTensor = None
    hidden_states: tuple[torch.FloatTensor] = None
    attentions: tuple[torch.FloatTensor] = None

# %% ../nbs/03_vitdet3dmae.ipynb 31
class ViTDet3DMAEPreTrainedModel(VitDet3dPreTrainedModel):
    config_class = ViTDet3DMAEConfig

    def _init_weights(self, module):
        super()._init_weights(module)

# %% ../nbs/03_vitdet3dmae.ipynb 32
class ViTDet3DMAEModel(ViTDet3DMAEPreTrainedModel):
    def __init__(self, config: ViTDet3DMAEConfig):
        super().__init__(config)

        self.config = config

        self.embeddings = ViTDet3DMAEEmbeddings(config)
        self.encoder = ViTDet3DMAEEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spacings: torch.FloatTensor,
        head_mask: torch.FloatTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, masks = self.embeddings(pixel_values, spacings)

        self.config.image_size = pixel_values.shape[2:]

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return ViTDet3DMAEModelOutput(
            last_hidden_state=sequence_output,
            masks=masks,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# %% ../nbs/03_vitdet3dmae.ipynb 36
@dataclass
class ViTDet3DMAEDecoderOutput(ModelOutput):
    logits: torch.FloatTensor
    hidden_states: tuple[torch.FloatTensor] = None
    attentions: tuple[torch.FloatTensor] = None

# %% ../nbs/03_vitdet3dmae.ipynb 37
class ViTDet3DMAEDecoder(nn.Module):
    def __init__(self, config: ViTDet3DMAEConfig):
        super().__init__()
        self.config = config

        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, config.decoder_hidden_size, 1, 1, 1))

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.num_channels * np.prod(config.patch_size), bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        spacings,
        masks,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # npZ etc. is the number of patches after uniform sampling
        # npD etc. is the number of patches in the original image
        # pZ etc. is the patch size
        # C is the number of channels in th eoriginal image
        # H can be any hidden size

        # embed tokens
        # (b, encoder_hidden_size, z, y, x)
        hidden_states = rearrange(hidden_states, "b h npZ npY npX -> b npZ npY npX h")
        x = self.decoder_embed(hidden_states)
        x = rearrange(x, "b npZ npY npX h -> b h npZ npY npX")
        # (b, decoder_hidden_size, z, y, x)

        # Create mask tokens array
        B, _, npZ, npY, npX = x.shape
        npD, npH, npW = npZ, npY, npX
        if self.config.mask_ratio == 0.875:
            npD, npH, npW = npZ * 2, npY * 2, npX * 2
        elif self.config.mask_ratio == 0.75:
            npD, npH, npW = npZ, npY * 2, npX * 2
        mask_tokens = self.mask_token.repeat(B, 1, npD, npH, npW)

        # Add hidden states to this
        x = mask_tokens.masked_scatter(masks.unsqueeze(1).bool(), x)

        # Get positional embeddings
        pD, pH, pW = self.config.patch_size
        grid_size, _ = self.config.get_grid_size_and_num_patches((npD * pD, npH * pH, npW * pW))
        position_embeddings = get_3d_position_embeddings(self.config.decoder_hidden_size, grid_size)
        position_embeddings = position_embeddings.to(x.device)
        # (1, hidden_size, z, y, x)
        if self.config.embed_spacing:
            position_embeddings = embed_spacings_in_position_embeddings(position_embeddings, spacings)
            # (b, hidden_size, z, y, x)

        # add pos embed
        hidden_states = x + position_embeddings

        # Flatten hidden states and reorder them
        hidden_states = rearrange(hidden_states, "b h npD npH npW -> b (npD npH npW) h")

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # Reshaping back to image patches
        logits = rearrange(logits, "b (npD npH npW) h -> b h npD npH npW", npD=npD, npH=npH, npW=npW)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)

        return ViTDet3DMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

# %% ../nbs/03_vitdet3dmae.ipynb 40
@dataclass
class ViTDet3DMAEForPreTrainingOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    masks: torch.LongTensor = None
    hidden_states: tuple[torch.FloatTensor] = None
    attentions: tuple[torch.FloatTensor] = None

# %% ../nbs/03_vitdet3dmae.ipynb 41
class ViTDet3DMAEForPreTraining(ViTDet3DMAEPreTrainedModel):
    def __init__(self, config: ViTDet3DMAEConfig):
        super().__init__(config)
        self.config = config

        self.vitdet3d = ViTDet3DMAEModel(config)
        self.decoder = ViTDet3DMAEDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def patchify(self, pixel_values):
        B, C, _, _, _ = pixel_values.shape
        grid_size, _ = self.config.get_grid_size_and_num_patches(pixel_values.shape[2:])
        npZ, npY, npX = grid_size
        pZ, pY, pX = self.config.patch_size

        target = rearrange(
            pixel_values,
            "b c (npZ pZ) (npY pY) (npX pX) -> b (c pZ pY pX) npZ npY npX",
            c=C,
            npZ=npZ,
            npY=npY,
            npX=npX,
            pZ=pZ,
            pY=pY,
            pX=pX,
        )

        return target

    def unpatchify(self, logits):
        C = self.config.num_channels
        B, _, npZ, npY, npX = logits.shape
        pZ, pY, pX = self.config.patch_size

        target = rearrange(
            logits,
            "b (c pZ pY pX) npZ npY npX -> b c (npZ pZ) (npY pY) (npX pX)",
            c=C,
            npZ=npZ,
            npY=npY,
            npX=npX,
            pZ=pZ,
            pY=pY,
            pX=pX,
        )

        return target

    def forward_loss(self, pixel_values, pred, masks_inverse):
        target = self.patchify(pixel_values)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=1)  # [N, L], mean loss per patch

        loss = (loss * masks_inverse).sum() / masks_inverse.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        spacings: torch.FloatTensor,
        head_mask: torch.FloatTensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vitdet3d(
            pixel_values,
            spacings,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        latent = outputs.last_hidden_state
        masks = outputs.masks

        decoder_outputs = self.decoder(latent, spacings, masks)
        logits = decoder_outputs.logits

        loss = self.forward_loss(pixel_values, logits, 1 - masks)

        if not return_dict:
            output = (logits, masks) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ViTDet3DMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            masks=masks,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
