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


## Example 
```python
import torch 
from medct.convnextv2 import ConvNextV2Config3d, ConvNextV2Model3d

config = ConvNextV2Config3d(dims=[40, 80], num_stages=2, num_channels=1, image_size=(96, 192, 192), depths=[3, 3])
model = ConvNextV2Model3d(config)
out = model(torch.randn((1, 1, 96, 192, 192)), output_hidden_states=True)
print([i.shape for i in out.hidden_states])
```
- out
```bash
[torch.Size([1, 40, 24, 48, 48]), 
 torch.Size([1, 40, 24, 48, 48]), 
 torch.Size([1, 80, 12, 24, 24])]
```

## Next steps
- Enable 3D Segmentation for all the networks 
- Enable 3D classification for all the networks
- Enbale MIM for swin transformers. 