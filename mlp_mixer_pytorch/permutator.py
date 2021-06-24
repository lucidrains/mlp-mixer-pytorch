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

def Permutator(*, image_size, patch_size, dim, depth, num_classes, segments, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
    height = width = image_size // patch_size
    s = segments

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                ParallelSum(
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b w c (h s)', s = s),
                        nn.Linear(height * s, height * s),
                        Rearrange('b w c (h s) -> b h w (c s)', s = s),
                    ),
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b h c (w s)', s = s),
                        nn.Linear(width * s, width * s),
                        Rearrange('b h c (w s) -> b h w (c s)', s = s),
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
