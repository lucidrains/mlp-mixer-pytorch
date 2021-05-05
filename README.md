<img src="./images/mlp-mixer.png" width="500px"></img>

## MLP Mixer - Pytorch

An All-MLP solution for Vision, from Google AI, in Pytorch.

No convolutions nor attention needed!

## Install

```bash
$ pip install mlp-mixer-pytorch
```

## Usage

```python
import torch
from mlp_mixer_pytorch import MLPMixer

model = MLPMixer(
    image_size = 256,
    patch_size = 16,
    dim = 512,
    depth = 12,
    num_classes = 1000
)

img = torch.randn(1, 3, 256, 256)
pred = model(img) # (1, 1000)
```

## Citations

```bibtex
@misc{tolstikhin2021mlpmixer,
    title   = {MLP-Mixer: An all-MLP Architecture for Vision},
    author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    year    = {2021},
    eprint  = {2105.01601},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
