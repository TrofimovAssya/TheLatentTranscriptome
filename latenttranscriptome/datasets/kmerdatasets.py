from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pdb
from collections import OrderedDict
import shutil
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import latenttranscriptome.utils.register as register


@register.setdatasetname("RandomKmerDataset")
class RandomKmerDataset(Dataset):

    def __init__(self, transform=None, nb_genes=20, kmer_length=24, nb_examples=1000, nb_base=4, nb_samples=5,
                 return_gene=False):

        self.transform = None
        self.nb_genes = nb_genes
        self.kmer_length = kmer_length
        self.nb_examples = nb_examples
        self.nb_base = nb_base
        self.nb_samples = nb_samples
        self.return_gene = return_gene
        self.process_data()

    def process_data(self):

        np.random.seed(1993)
        def ngrams_via_striding(array, order):
            itemsize = array.itemsize
            assert array.strides == (itemsize,)
            return as_strided(array, (max(array.size + 1 - order, 0), order), (itemsize, itemsize))

        print("Striding the data...")

        self.data = np.random.randint(0, self.nb_base, ((1 + self.nb_examples) * self.kmer_length,))
        self.data = ngrams_via_striding(self.data, self.kmer_length)
        self.nb_examples = self.data.shape[0]  # Not exactly the same
        gene_association = np.random.randint(0, self.nb_genes, (self.nb_examples))
        gene_association = np.array(sorted(gene_association))
        self.gene_association = gene_association
        self.gene_distributions = []

        # for every samples, get a count
        print("Computing the targets...")
        targets = []
        patients = []
        for i in range(self.nb_samples):
            gene_distribution = np.random.dirichlet([1] * self.nb_genes, 1)  # This dude gene distribution
            self.gene_distributions.append(gene_distribution)
            # gene_distribution = range(self.nb_genes)
            counts = self.compute_one_sample(gene_distribution, gene_association)
            patient_id = [i] * len(counts)

            targets.append(counts)
            patients.append(patient_id)

        self.targets = np.array(targets).reshape(-1)
        self.patients = np.array(patients).reshape(-1)
        print(self.patients.sum(), len(self.patients))
        print("Done!")

    def compute_one_sample(self, gene_distribution, gene_association):

        target = np.zeros((self.nb_examples, self.nb_genes))
        for i in range(self.nb_genes):
            target[gene_association == i, i] = 1

        target = (target * gene_distribution).sum(axis=1)
        return target

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        sample = self.data[idx % self.nb_examples]
        label = self.targets[idx]

        sample = np.pad(sample, ((0, 1),), 'constant', constant_values=(self.patients[idx],))

        if self.return_gene:
            sample = [sample, label, self.gene_association[idx % self.nb_examples]]
        else:
            sample = [sample, label]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def input_size(self):

        # 4 for [A, C, G, T]
        # We have only one patient right now. (TODO: add more patient)

        return self.nb_base, self.nb_samples

    def extra_info(self, input_no):
        # get some extra info to dumb with the embeddings.
        info = OrderedDict()

        if input_no == 0:
            info['base'] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        return info