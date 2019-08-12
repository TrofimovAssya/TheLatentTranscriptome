#!/usr/bin/env python
import torch
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
import models
import pickle
import time
import monitoring
#
def build_parser():
    parser = argparse.ArgumentParser(description="")

    ### Hyperparameter options
    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=1, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['kmer'], default='kmer', help='Which dataset to use.')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    parser.add_argument('--nb-patient', default=5,type=int, help='nb of different patients')
    parser.add_argument('--nb-kmer', default=1000,type=int, help='nb of different kmers')
    # Model specific options
    parser.add_argument('--layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb_size', default=2, type=int, help='The size of the embeddings.')
    parser.add_argument('--loss', choices=['NLL', 'MSE'], default = 'MSE', help='The cost function to use')

    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['RNN'], default='RNN', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Monitoring options
    parser.add_argument('--save-error', action='store_true', help='If we want to save the error for each tissue and each gene at every epoch.')
    parser.add_argument('--make-grid', default=True, type=bool,  help='If we want to generate fake patients on a meshgrid accross the patient embedding space')
    parser.add_argument('--nb-gridpoints', default=50, type=int, help='Number of points on each side of the meshgrid')
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    # TODO: set the seed
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None: # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    # creating the dataset
    print ("Getting the dataset...")
    dataset = datasets.get_dataset(opt,exp_dir)

    # Creating a model
    print ("Getting the model...")
    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), )

    criterion = torch.nn.MSELoss()
    # Training optimizer and stuff
    if opt.loss == 'NLL':
        criterion = torch.nn.NLLLoss()


    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")
    #monitoring and predictions
    predictions =np.zeros((dataset.dataset.nb_patient,dataset.dataset.nb_kmer))
    indices_patients = np.arange(dataset.dataset.nb_patient)
    indices_genes = np.arange(dataset.dataset.nb_kmer)
    xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                          np.repeat(indices_patients, len(indices_genes))])
    progress_bar_modulo = len(dataset)/100
    for t in range(epoch, opt.epoch):

        start_timer = time.time()

        if opt.save_error:
            outfname_g = '_'.join(['gene_epoch',str(t),'prediction.npy'])
            outfname_g = os.path.join(exp_dir,outfname_g)
            outfname_t = '_'.join(['tissue_epoch',str(t),'prediction.npy'])
            outfname_t = os.path.join(exp_dir,outfname_t)
            train_trace = np.zeros((dataset.dataset.nb_gene, dataset.dataset.nb_patient))

        for no_b, mini in enumerate(dataset):

            inputs_s, inputs_k, targets = mini[0], mini[1], mini[2]

            inputs_s = Variable(inputs_s, requires_grad=False).float()
            inputs_k = Variable(inputs_k, requires_grad=False).float()
            targets = Variable(targets, requires_grad=False).float()

            if not opt.cpu:
                inputs_s = inputs_s.cuda(opt.gpu_selection)
                inputs_k = inputs_k.cuda(opt.gpu_selection)
                targets = targets.cuda(opt.gpu_selection)

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = my_model(inputs_k,inputs_s).float()

            targets = torch.reshape(targets,(targets.shape[0],1))
            # Compute and print loss

            loss = criterion(y_pred, targets)
            if no_b % 5 == 0:
                print (f"Doing epoch {t},examples{no_b}/{len(dataset)}.Loss:{loss.data.cpu().numpy().reshape(1,)[0]}")

                # Saving the emb
                np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy())


            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print ("Saving the model...")
        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)


if __name__ == '__main__':
    main()
