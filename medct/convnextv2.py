# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_convnextv2.ipynb.

# %% auto 0
__all__ = ['ConvNextV2GRN3d', 'ConvNextV2LayerNorm3d', 'ConvNextV2Embeddings3d', 'ConvNextV2Layer3d', 'ConvNextV2Stage3d',
           'ConvNextV2Encoder3d', 'ConvNextV2Config3d', 'ConvNextV2PreTrainedModel3d', 'ConvNextV2Model3d',
           'ConvNextV2Backbone3d']

# %% ../nbs/00_convnextv2.ipynb 2
import torch
import torch.nn as nn
import fastcore.all as fc

from typing import Optional, Union, Tuple
from transformers.models.convnextv2.modeling_convnextv2 import drop_path, ConvNextV2DropPath, ConvNextV2LayerNorm, BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
from transformers.modeling_outputs import BackboneOutput
from transformers.utils.backbone_utils import BackboneMixin

# %% ../nbs/00_convnextv2.ipynb 16
class ConvNextV2GRN3d(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # Compute and normalize global spatial feature maps - N, H, W, D, C
        global_features = torch.norm(hidden_states, p=2, dim=(1, 2, 3), keepdim=True)
        norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        return hidden_states

# %% ../nbs/00_convnextv2.ipynb 20
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2LayerNorm 
class ConvNextV2LayerNorm3d(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None] #Only this line changed
        return x

# %% ../nbs/00_convnextv2.ipynb 26
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2Embeddings with 
# ConvNextV2Embeddings -> ConvNextV2Embeddings3D
class ConvNextV2Embeddings3d(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextV2LayerNorm3d(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.layernorm(embeddings)
        return embeddings

# %% ../nbs/00_convnextv2.ipynb 31
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2Layer 
class ConvNextV2Layer3d(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, D, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config3D`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = ConvNextV2LayerNorm3d(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = ACT2FN[config.hidden_act]
        self.grn = ConvNextV2GRN3d(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        x = self.dwconv(hidden_states)
        # (batch_size, num_channels, height, width, deoth) -> (batch_size, height, width, depth, num_channels)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # (batch_size, height, width, depth, num_channels) -> (batch_size, num_channels, height, width, depth)
        x = x.permute(0, 4, 1, 2, 3)

        x = input + self.drop_path(x)
        return x

# %% ../nbs/00_convnextv2.ipynb 36
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2Stage 
class ConvNextV2Stage3d(nn.Module):
    """ConvNeXTV23D stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextV2Config3D`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()

        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextV2LayerNorm3d(in_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            self.downsampling_layer = nn.Identity()
        drop_path_rates = drop_path_rates or [0.0] * depth
        self.layers = nn.Sequential(
            *[ConvNextV2Layer3d(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)
        hidden_states = self.layers(hidden_states)
        return hidden_states

# %% ../nbs/00_convnextv2.ipynb 38
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2Encoder 
class ConvNextV2Encoder3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextV2Stage3d(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )



# %% ../nbs/00_convnextv2.ipynb 39
# Copied from transformers.models.convnextv2.configuration_convnextv2.ConvNextV2Config
class ConvNextV2Config3d(BackboneConfigMixin, PretrainedConfig):
    model_type = "convnextv23d"

    def __init__(
        self,
        num_channels=3,
        patch_size=4,
        num_stages=2,
        hidden_sizes=None,
        depths=None,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        drop_path_rate=0.0,
        image_size=224,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_stages = num_stages
        self.hidden_sizes = [40, 80] if hidden_sizes is None else hidden_sizes
        self.depths = [3, 3] if depths is None else depths
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.drop_path_rate = drop_path_rate
        self.image_size = image_size
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )

# %% ../nbs/00_convnextv2.ipynb 44
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2PreTrainedModel 
class ConvNextV2PreTrainedModel3d(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextV2Config3d
    base_model_prefix = "convnextv2_3d"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# %% ../nbs/00_convnextv2.ipynb 46
# Copied from transformers.models.convnextv2.modeling_convnextv2.ConvNextV2Model 
class ConvNextV2Model3d(ConvNextV2PreTrainedModel3d):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ConvNextV2Embeddings3d(config)
        self.encoder = ConvNextV2Encoder3d(config)

        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # global average pooling, (N, C, H, W, D) -> (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-3, -2, -1]))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )

# %% ../nbs/00_convnextv2.ipynb 54
# Copied from transformers.models.convnext.modeling_convnext.ConvNextBackbone with CONVNEXT->CONVNEXTV2,ConvNext->ConvNextV2,facebook/convnext-tiny-224->facebook/convnextv2-tiny-1k-224
class ConvNextV2Backbone3d(ConvNextV2PreTrainedModel3d, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        self.embeddings = ConvNextV2Embeddings3d(config)
        self.encoder = ConvNextV2Encoder3d(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # Add layer norms to hidden states of out_features
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm3d(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        feature_maps = ()
        # we skip the stem
        for idx, (stage, hidden_state) in enumerate(zip(self.stage_names[1:], hidden_states[1:])):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
