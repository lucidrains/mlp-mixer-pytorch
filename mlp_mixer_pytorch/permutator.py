from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))

def Permutator(*, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    height = width = image_size // patch_size
    num_patches = (height * width) ** 2

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                ParallelSum(
                    nn.Sequential(
                        Rearrange('b h w c -> b w c h'),
                        nn.Linear(height, height),
                        Rearrange('b w c h -> b h w c'),
                    ),
                    nn.Sequential(
                        Rearrange('b h w c -> b h c w'),
                        nn.Linear(width, width),
                        Rearrange('b h c w -> b h w c'),
                    ),
                    nn.Linear(dim, dim)
                ),
                nn.Linear(dim, dim)
            )),
            PreNormResidual(dim, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * expansion_factor, dim),
                nn.Dropout(dropout)
            ))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b h w c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )
