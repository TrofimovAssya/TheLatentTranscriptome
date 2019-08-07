from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pdb
from collections import OrderedDict
import shutil
import pandas as pd


class KmerDataset(Dataset):
    """Kmer abundance dataset"""

    def __init__(self,root_dir='.',save_dir='.',data_file='data.npy'):


        data_path = os.path.join(root_dir, data_file)
        self.data = pd.read_csv(data_path,index_col=0)

        self.nb_patient = self.data.shape[0]
        self.nb_kmer = self.data.shape[1]
        print (self.nb_kmer)
        print (self.nb_patient)

        self.root_dir = root_dir
        self.X_data, self.Y_data = self.dataset_make(np.array(self.data),log_transform=False)
        self.X_sample = self.X_data[:,1]
        self.X_kmer = self.data.columns[self.X_data[:,0]]
        self.X_kmer = self.transform_kmerseq_table(self.X_kmer)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):

        sample = self.X_sample[idx]
        kmer = self.X_kmer[idx]
        label = self.Y_data[idx]

        sample = [sample, kmer, label]

        return sample

    def dataset_make(self, gene_exp, log_transform=False):
        indices_p1 = np.arange(gene_exp.shape[0])
        indices_g = np.arange(gene_exp.shape[1])
        X_data = np.transpose([np.tile(indices_g, len(indices_p1)), np.repeat(indices_p1, len(indices_g))])
        Y_data = gene_exp[X_data[:, 1], X_data[:, 0]]

        print (f"Total number of examples: {Y_data.shape} ")

        if log_transform:
            Y_data = np.log10(Y_data + 1)
        return X_data, Y_data

    def get_kmer_onehot(self,kmer):
        convert = {'A':3, 'G':1, 'C':2, 'T':0}
        result = np.zeros((len(kmer), 4))
        for i in range(len(kmer)):
            result[i,convert[kmer[i]]] +=1
        return result

    def transform_kmerseq_table(self, X_kmer):
        X_kmer = list(X_kmer)
        out_kmers = np.zeros((len(X_kmer), len(X_kmer[0])  , 4 ))

        for kmer in X_kmer:
            out_kmers[X_kmer.index(kmer)] = self.get_kmer_onehot(kmer)
        return np.array(out_kmers)

    def input_size(self):
        return self.nb_patient, self.nb_kmer

    def extra_info(self):
        info = OrderedDict()
        return info



def get_dataset(opt, exp_dir):

    if opt.dataset == 'kmer':
        dataset = KmerDataset(root_dir=opt.data_dir, save_dir =exp_dir,data_file = opt.data_file)
    else:
        raise NotImplementedError()

    #TODO: check the num_worker, might be important later on, for when we will use a bunch of big files.
    dataloader = DataLoader(dataset, batch_size=opt.batch_size,shuffle=True,num_workers=1)
    return dataloader
