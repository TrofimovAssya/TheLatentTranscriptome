from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import pdb
from collections import OrderedDict
import shutil
import pandas as pd
from numpy.lib.stride_tricks import as_strided
import factembseq.utils.register as register


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


# -------------------------------------
# Bellow is the old datasets. Keeping them here for now, but I'll delete them or register them later
# -------------------------------------
#
class KmerDataset(Dataset):

    def __init__(self, root_dir='/data/milatmp1/dutilfra/dataset/kmer/', data_path='duodenum1.24.hdf5',
                 transform=None):

        data_path = os.path.join(root_dir, data_path)

        # Load the dataset
        self.data = h5py.File(data_path)['kmer']

        self.nb_kmer = self.data.shape[0]
        self.nb_tissue = 1  # TODO
        self.nb_patient = 1  # TODO

        self.root_dir = root_dir
        self.transform = transform  # heh

    def __len__(self):
        return self.nb_kmer

    def __getitem__(self, idx):

        sample = self.data[idx, :-1]
        label = self.data[idx, -1]
        label = np.log(label)
        sample = np.pad(sample, ((0, 1),), 'constant', constant_values=(0,))  # adding the patient TODO: the real one.

        sample = [sample, label]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def input_size(self):

        # 4 for [A, C, G, T]
        # We have only one patient right now. (TODO: add more patient)

        return 4, self.nb_patient

    def extra_info(self, input_no):
        # get some extra info to dumb with the embeddings.
        info = OrderedDict()

        if input_no == 0:
            info['base'] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        return info


class GeneralKmerDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        # Load the dataset
        self.data = []
        self.nb_gene_expressions_per_patient = [0]

        for file in os.listdir(root_dir):
            file = os.path.join(root_dir, file)

            data = h5py.File(file)['kmer']
            self.data.append(data)
            self.nb_gene_expressions_per_patient.append(data.shape[0] + self.nb_gene_expressions_per_patient[-1])

        print
        "We have {} patients.".format(len(self.data))

        self.nb_kmer = sum([data.shape[0] for data in self.data])
        self.nb_tissue = 1  # TODO
        self.nb_patient = len(self.nb_gene_expressions_per_patient) - 1  # TODO

        self.root_dir = root_dir
        self.transform = transform  # heh

    def __getitem__(self, idx):

        # Get the corresponding bin.
        patient_id = np.digitize([idx], self.nb_gene_expressions_per_patient[1:])[0]  # TODO: Do for tissue as well?
        gene_id = idx - self.nb_gene_expressions_per_patient[patient_id]

        # Get the correct patient
        sample = np.array(self.data[patient_id][gene_id, :-1])
        label = np.array(self.data[patient_id][gene_id, -1])

        label = np.log(label)
        sample = np.pad(sample, ((0, 1),), 'constant', constant_values=(patient_id,))  # adding the patient

        sample = [sample, label]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.nb_kmer

    def input_size(self):

        # 4 for [A, C, G, T]
        # We have only one patient right now.
        return 4, self.nb_patient

    def extra_info(self, input_no):
        # get some extra info to dumb with the embeddings.
        info = OrderedDict()

        if input_no == 0:
            info['base'] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        return info


class GeneKmerDataset(Dataset):
    def __init__(self, gene_file="data/xist.txt"):
        self.transform = None
        self.nb_genes = 1
        self.kmer_length = 24
        self.gene_file = gene_file
        self.process_data()

    def process_data(self):
        gene = open(self.gene_file).read()
        gene = gene.replace(' ', '').replace('\n', '').replace('a', '0').replace('c', '1').replace('g', '2').replace(
            't', '3')

        from numpy.lib.stride_tricks import as_strided

        def ngrams_via_striding(array, order):
            itemsize = array.itemsize
            assert array.strides == (itemsize,)
            return as_strided(array, (max(array.size + 1 - order, 0), order), (itemsize, itemsize))

        self.gene_kmer = ngrams_via_striding(np.array(list(gene), dtype=int), self.kmer_length)

    def __len__(self):
        return len(self.gene_kmer)

    def __getitem__(self, idx):
        sample = self.gene_kmer[idx]
        label = idx

        sample = np.pad(sample, ((0, 1),), 'constant', constant_values=(0,))
        sample = [sample, label]
        return sample

    def input_size(self):
        # 4 for [A, C, G, T]
        # We have only one patient right now. (TODO: add more patient)

        return 4, 18

    def extra_info(self, input_no):
        # get some extra info to dumb with the embeddings.
        info = OrderedDict()

        if input_no == 0:
            info['base'] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        return info


class FewGeneKmerDataset(Dataset):

    def __init__(self, root_dir='/data/lisa/data/genomics/GTEx/',
                 file_name='uterus/total_uterus_kmers.pkl', transform=None, normalize='sum', min_count=0):

        # Load the dataset
        self.data = {}
        # self.count = []
        # self.nb_gene_expressions_per_folder = [0]
        # self.sample = []

        file = os.path.join(root_dir, file_name)

        fl = pd.read_pickle(file)
        self.data = fl
        self.nb_tissue = False
        # data = fl['kmer']
        # data = {c: fl[c] for c in columns}

        self.data = self.data.loc[self.data['count'] > min_count]  # removing the non-common kmers.

        if normalize == 'sum':
            print
            "Normalizing by the sum of each sample."
            self.data['count'] = fl['count'] / fl.groupby('sample')['count'].transform(np.sum)  # normalize by the sum
        elif normalize == 'max':
            print
            "Normalizing by the max of each sample."
            self.data['count'] = fl['count'] / fl.groupby('sample')['count'].transform(np.max)  # normalize by the max
        elif normalize == 'log':
            self.data['count'] = np.log(fl['count'] + 1.)  # normalize by log
        elif normalize == 'sum-log':
            self.data['count'] = np.log(fl['count'] + 1.)  # normalize by log
            self.data['count'] = self.data['count'] / self.data.groupby('sample')['count'].transform(
                np.sum)  # normalize by the sum
        elif normalize == 'center-log':
            self.data['count'] = np.log(fl['count'] + 1.)  # normalize by log
            self.data['count'] = (
                        self.data['count'] - self.data.groupby('sample')['count'].transform(np.mean))  # center
        elif normalize == 'norm-log':
            self.data['count'] = np.log(fl['count'] + 1.)  # normalize by log
            self.data['count'] = (self.data['count'] - self.data.groupby('sample')['count'].transform(
                np.mean)) / np.sqrt(self.data.groupby(['sample'])['count'].transform(np.var))  # normalized

        self.data['sample'] = fl['sample'].astype('category')

        if 'tissue' in fl.keys():
            self.data['tissue'] = fl['tissue'].astype('category')
            self.nb_tissue = len(self.data['tissue'].cat.categories)
        else:
            self.nb_tissue = 1
        # self.data.append(data)
        # self.count.append(count)
        # self.sample.append(sample)
        # self.nb_gene_expressions_per_folder.append(len(data) + self.nb_gene_expressions_per_folder[-1])
        # print "We have {} files.".format(len(self.data))
        # self.nb_kmer = sum([data.shape[0] for data in self.data])
        # self.nb_tissue = len(self.data)
        # self.nb_patient = sum([len(set(x)) for x in self.sample])

        self.nb_kmer = len(self.data['kmer'])
        self.nb_patient = len(self.data['sample'].cat.categories)

        print
        "We have {} kmers.".format(self.nb_kmer)

        self.root_dir = root_dir
        self.transform = transform  # heh
        self.filtered_data = self.data

    # We can set the iterator to filter the kmers.
    def set_filter(self, filter=''):
        if filter:
            self.filtered_data = self.data.query(filter)
        else:
            self.filtered_data = self.data

    def __getitem__(self, idx):

        # Get the corresponding bin.

        # Get the correct patient
        # sample = np.array(self.filtered_data['kmer'].loc[self.filtered_data.index[idx]], dtype=float)
        sample = np.array(map(float, self.filtered_data['kmer'].loc[self.filtered_data.index[idx]]))
        label = np.array(self.filtered_data['count'].loc[self.filtered_data.index[idx]])
        sample_id = self.filtered_data['sample'].cat.codes[self.filtered_data.index[idx]]

        sample = np.pad(sample, ((0, 1),), 'constant',
                        constant_values=(sample_id,))  # adding the patient

        if self.nb_tissue > 1:
            tissue_id = self.data['tissue'].cat.codes[idx]
            sample = np.pad(sample, ((0, 1),), 'constant',
                            constant_values=(tissue_id,))  # adding the tissue

        sample = [sample, np.array([label])]

        if self.transform:
            sample = self.transform(sample)

        # print sample
        return sample

    def __len__(self):
        return len(self.filtered_data)

    def input_size(self):

        # 4 for [A, C, G, T]
        if self.nb_tissue == 1:
            return 4, self.nb_patient
        else:
            return 4, self.nb_patient, self.nb_tissue

    def extra_info(self, input_no):
        # get some extra info to dumb with the embeddings.
        info = OrderedDict()

        if input_no == 0:
            info['base'] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        return info


class FewGeneKmerDatasetTissue(FewGeneKmerDataset):

    def __init__(self, file_name='zfy_myh6_uterus_heart.pkl', **kwargs):
        super(FewGeneKmerDatasetTissue, self).__init__(file_name=file_name, **kwargs)


class FewGeneKmerDataset1517(FewGeneKmerDataset):
    def __init__(self, root_dir='/data/lisa/data/genomics/AML/',
                 file_name='6patients_AML_total_kmers_t1517.pkl', **kwargs):
        super(FewGeneKmerDataset1517, self).__init__(file_name=file_name,
                                                     root_dir=root_dir, **kwargs)


class FewGeneKmerDataset1517_149samples(FewGeneKmerDataset):
    def __init__(self, root_dir='/data/lisa/data/genomics/AML/',
                 file_name='AML_total_kmers_t1517_filter.pkl', **kwargs):
        super(FewGeneKmerDataset1517_149samples, self).__init__(file_name=file_name,
                                                                root_dir=root_dir, **kwargs)


class FewGeneKmerDatasetZFXZFY_6samples(FewGeneKmerDataset):
    def __init__(self, root_dir='/data/lisa/data/genomics/AML/',
                 file_name='6patient_kmers_zfy_filter.pkl', **kwargs):
        super(FewGeneKmerDatasetZFXZFY_6samples, self).__init__(file_name=file_name,
                                                                root_dir=root_dir, **kwargs)
# A list of specific dataset
class KmerHeartAtrialDataset(GeneralKmerDataset):

    def __init__(self, root_dir='/data/lisa/data/genomics/GTEx/heart_atrial/hdf5/', transform=None):
        super(KmerHeartAtrialDataset, self).__init__(root_dir=root_dir, transform=transform)


class HeartAtrialUterusDataset(GeneralKmerDataset):

    def __init__(self, root_dir='/data/lisa/data/genomics/GTEx/hdf5_convert/', transform=None):
        super(HeartAtrialUterusDataset, self).__init__(root_dir=root_dir, transform=transform)


