import torch
import torch.nn.functional as F
from torch import nn

class FactorizedMLP(nn.Module):

    def __init__(self, layers_size, inputs_size, emb_size=2, nb_emb=2):
        super(FactorizedMLP, self).__init__()

        if type(emb_size) != list:
            emb_size = [emb_size] * len(inputs_size)

        self.layers_size = layers_size
        self.emb_size = emb_size
        self.inputs_size = inputs_size
        self.nb_emb = nb_emb

        self.embs = nn.ModuleList([nn.Embedding(inputs_size[i], emb_size[i]) for i in range(nb_emb)])

        # The list of layers.
        layers = []
        dim = [self.get_embeddings_size()] + layers_size # Adding the emb size.
        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 1)

    def get_embeddings_size(self):
        return sum(self.emb_size)

    def get_embeddings(self, x):

        embs = []
        for i in range(self.nb_emb):
            embs.append(self.embs[i](x[:, i].long()))

        return embs

    def forward(self, x):

        # Get the embeddings
        embs = self.get_embeddings(x)

        # Forward pass.
        mlp_input = torch.cat(embs, 1)

        # TODO: the proper way in pytorch is to use a Sequence layer.
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = torch.tanh(mlp_input)

        mlp_output = self.last_layer(mlp_input)

        return mlp_output

    def get_emb_params(self):
        return self.embs.parameters()

    def get_not_emb_params(self):

        emb_params = set(self.get_emb_params())
        all_params = set(self.parameters())

        other_params = all_params - emb_params

        return iter(other_params)
