from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pdb
from collections import OrderedDict
import shutil
import pandas as pd
import factembseq.utils.register as register


@register.setdatasetname("GeneDataset")
class GeneDataset(Dataset):
    """Gene expression dataset"""

    def __init__(self, root_dir='.', data_path='30by30_dataset.npy', data_type_path='30by30_types.npy', data_subtype='30by30_subtypes.npy', transform=None):


        data_path = os.path.join(root_dir, data_path)
        data_type_path = os.path.join(root_dir, data_type_path)
        data_subtype = os.path.join(root_dir, data_subtype)

        # Load the dataset
        self.data, self.data_type, self.data_subtype = np.load(data_path), np.load(data_type_path), np.load(data_subtype)

        self.nb_gene = self.data.shape[0]
        self.nb_tissue = len(set(self.data_type))
        self.nb_patient = self.data.shape[1]

        # TODO: for proper pytorch form, we should probably do that on the fly, but heh. todo future me.
        self.X_data, self.Y_data = self.dataset_make(self.data, log_transform=True)
        #self.X_data = self.X_data[:1000]
        #self.Y_data = self.Y_data[:1000]

        #import ipdb; ipdb.set_trace()

        self.root_dir = root_dir
        self.transform = transform # heh

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_data[idx]
        label = self.Y_data[idx]

        sample = [sample, label]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def dataset_make(self, gene_exp, log_transform=True):

        #    indices_p1 = numpy.random.randint(0, gene_exp.shape[1]-1,nb_examples)
        indices_p1 = np.arange(gene_exp.shape[1])
        indices_g = np.arange(gene_exp.shape[0])
        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        Y_data = gene_exp[X_data[:, 0], X_data[:, 1]]

        print("Total number of examples: ", Y_data.shape)

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def input_size(self):
        return self.nb_gene, self.nb_patient

    def extra_info(self, input_no):
        # get some extra info to dumb with the embeddings.
        info = OrderedDict()

        if input_no == 1:
            info['type'] = self.data_type
            info['subtype'] = self.data_subtype

        return info

