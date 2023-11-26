# medct
3D networks to support MRI and CT 


## Setup 
- To remove the environment do `conda env remove -n medct`
- To create an enviroment `conda create -n medct python=3.9`
- `conda activate medct` & do `pip install -e .`
Developers setup kernel for running notebooks using `python -m ipykernel install --user --name medct --display-name "Python (medct)"`
clear jupyter notebook outputs jupyter nbconvert --clear-output --inplace *.ipynb


## Updates 
- [11-Nov, 2023] ConvNextV23d encoder
- [24-Nov, 2023] Swin3D encoder 
- [26-Nov, 2023] Swin3D masked image modelling
- [26-Nov, 2023] ConvNextV2 masked image modelling


## Example - ConvNextV2Backbone
```python
import torch 
from medct.convnextv2 import ConvNextV2Config3d, ConvNextV2Backbone3d

config = ConvNextV2Config3d(dims=[40, 80], num_stages=2, num_channels=1, image_size=(96, 192, 192), depths=[3, 3])
model = ConvNextV2Backbone3d(config)
out = model(torch.randn((1, 1, 96, 192, 192)), output_hidden_states=True)
print([i.shape for i in out.feature_maps])
```
- out
```bash
[torch.Size([2, 40, 48, 48, 48]), 
 torch.Size([2, 80, 24, 24, 24])]
```

## Example - Convnextv2mim
```python
import torch 
from medct.convnextv2 import ConvNextV2Config3d
from medct.swin3dmim import ConvNextV2ForMaskedImageModeling, mask_patches
config = ConvNextV2Config3d(num_channels=1, image_size=(96, 192, 192), patch_size=(8, 16, 16), hidden_sizes=[40, 80], depths=[2, 2])
model = ConvNextV2ForMaskedImageModeling(config)
bool_masked_pos = mask_patches(model.num_patches, 0.4)
out = model(torch.randn((1, 1, )+model.config.image_size), bool_masked_pos=bool_masked_pos)
print(out.loss)
```

## Example - Swin3DBackbone
```python
import torch 
from medct.swin3d import Swin3dBackbone, Swin3dConfig
config = Swin3dConfig(embed_dim=96, depths=(2, 4), out_features=["stage1", "stage2"])
model = Swin3dBackbone(config)
out = model(torch.randn((1, 1, 96, 192, 192)), output_hidden_states=True)
print([i.shape for i in out.feature_maps])
```
- out
```bash
[torch.Size([1, 96, 48, 48, 48]),
 torch.Size([1, 192, 24, 24, 24])
 ]
```

## Example - SwinMIM
```python
import torch 
from medct.swin3d import Swin3dConfig
from medct.swin3dmim import Swin3dForMaskedImageModeling, mask_patches
config = Swin3dConfig(image_size=(96, 192, 192), depths=[2, 2], num_heads=[3, 6], patch_size=(8, 16, 16), encoder_stride=(16, 32, 32))
model = Swin3dForMaskedImageModeling(config)
bool_masked_pos = mask_patches(model.num_patches, 0.4)
out = model(torch.randn((1, 1, )+model.config.image_size), bool_masked_pos=bool_masked_pos)
print(out.loss)
```
