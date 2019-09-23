import torch
import torch.nn.functional as F
from torch import nn
import argparse
from itertools import chain


class FactorizedRNN(nn.Module):

    def __init__(self, layers_size=[10,2], nb_samples=1, emb_size=2, data_dir ='.'):
        super(FactorizedRNN, self).__init__()

        self.emb_size = emb_size
        self.sample = nb_samples
        
        #2 spaces: emb1 = samples and emb2 = kmer
        # input_size is the 4 nucleotides as a one-hot
        # hidden_size is the RNN output
        # this rnn gives the kmer embedding
        #self.rnn = nn.LSTM(input_size=4 , hidden_size=2, bias=True,
        #                          batch_first=True, bidirectional=False, num_layers=1)
        
        self.conv1 = nn.Conv1d(in_channels = 4,out_channels = 1,kernel_size = 3,stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 1,out_channels = 1,kernel_size = 3,stride = 2)
        self.conv3 = nn.Conv1d(in_channels = 1,out_channels = 1,kernel_size = 3,stride = 7)

        self.emb_1 = nn.Embedding(self.sample, emb_size)
        

        layers = []
        dim = [emb_size + 2] + layers_size # Adding the emb size. (2d embeddings for the kmers)

        for size_in, size_out in zip(dim[:-1], dim[1:]):
            layer = nn.Linear(size_in, size_out)
            layers.append(layer)

        self.mlp_layers = nn.ModuleList(layers)

        # Last layer
        self.last_layer = nn.Linear(dim[-1], 1)


    def get_embeddings(self, x1, x2):

        kmer, patient = x1, x2
        # Kmer Embedding.
        #_, (h, c) = self.rnn(kmer)
        #kmer = h.squeeze() # get rid of extra dimension
        kmer = self.conv1(kmer)
        kmer = self.conv2(kmer)
        kmer = self.conv3(kmer)
        kemr = kmer.squeeze()
        # Patient Embedding
        patient = self.emb_1(patient.long())        
        return kmer, patient

    def forward(self, x1, x2):

        # Get the embeddings
        emb_1, emb_2 = self.get_embeddings(x1, x2)
        emb_2 = emb_2.view(-1,2)
        if not emb_1.shape == emb_2.shape:
            import pdb; pdb.set_trace()
        mlp_input = torch.cat([emb_1, emb_2], 1)
        # Forward pass.
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
            mlp_input = F.tanh(mlp_input)
        
        
        mlp_output = self.last_layer(mlp_input)

        return mlp_output




def get_model(opt, inputs_size, model_state=None):

    if opt.model == 'RNN':
        model_class = FactorizedRNN
        model = model_class(layers_size=opt.layers_size, nb_samples=inputs_size[0], emb_size=opt.emb_size, data_dir = opt.data_dir)
    else:
        raise NotImplementedError()

    if model_state is not None:
        model.load_state_dict(model_state)

    return model
