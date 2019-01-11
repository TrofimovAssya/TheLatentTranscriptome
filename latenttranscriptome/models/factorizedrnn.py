import torch
import torch.nn.functional as F
from torch import nn
import latenttranscriptome.utils.register as register
from latenttranscriptome.models.factorizedmlp import FactorizedMLP

@register.setmodelname("FactorizedRNN")
class FactorizedRNN(FactorizedMLP):
    '''
    RNN approach. Each kmers is a word. We pass a rnn on top of them.
    '''

    def __init__(self, input_emb_size=2,
                 num_layers=1,
                 bidirectional=True,
                 hidden_size=256,
                 emb_size=2,
                 nb_emb=1,
                 **kwargs):

        emb_size = [input_emb_size] +  [emb_size] * (nb_emb - 1) # Changing the embeddings size.
        kwargs['emb_size'] = emb_size
        kwargs['nb_emb'] = nb_emb

        self.emb_size = emb_size
        self.input_emb_size = input_emb_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.nb_emb = nb_emb

        super(FactorizedRNN, self).__init__(**kwargs)

        dims = [input_emb_size] + [hidden_size] * num_layers

        # The normal layers
        self.rnn = nn.LSTM(input_size=input_emb_size, hidden_size=hidden_size, bias=True,
                                  batch_first=True, bidirectional=bidirectional, num_layers=num_layers)

        #The last layer
        self.mlp = nn.Linear(dims[-1] * num_layers * (int(bidirectional + 1)), emb_size[-1])

    def get_embeddings_size(self):
        return self.emb_size[-1] * self.nb_emb

    def get_embeddings(self, x):

        # Splitting the embeddings.
        kmer, rest = x[:, :-(self.nb_emb-1)], x[:, -(self.nb_emb-1):]

        # Getting the kmer embedding.
        kmer = self.embs[0](kmer.squeeze(-1).long())
        kmer = self.rnn(kmer)[1][0] # Get the last output
        kmer = kmer.permute(1, 0, 2)
        kmer = kmer.contiguous().view(kmer.size(0), -1)
        kmer = self.mlp(kmer)

        # Getting the other embedding.
        embs = [kmer]
        for i in range(1, self.nb_emb):
            embs.append(self.embs[i](rest[:, i-1].long()))

        return embs
