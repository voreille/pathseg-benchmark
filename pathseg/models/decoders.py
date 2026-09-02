from torch import nn


class LinearSemanticDecoder(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, feature_maps, task=None):
        x = feature_maps[-1]

        if task is not None:
            return {task: self.heads[task](x)}

        return {name: head(x) for name, head in self.heads.items()}
