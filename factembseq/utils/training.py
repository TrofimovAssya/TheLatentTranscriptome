from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import torch
import numpy as np
import random
from collections import OrderedDict
import copy
import factembseq.utils.configuration as configuration
import factembseq.utils.monitoring as monitoring

_LOG = logging.getLogger(__name__)

def train(cfg):

    print("Our config:", cfg)
    seed = cfg['seed']
    cuda = cfg['cuda']
    num_epoch = cfg['epoch']
    device = 'cuda' if cuda else 'cpu'

    # Setting the seed.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if cuda:
        torch.cuda.manual_seed_all(seed)

    # Dataset
    # transform
    tr_train = configuration.setup_transform(cfg, 'train')
    tr_valid = configuration.setup_transform(cfg, 'valid')

    # The dataset
    dataset_train = configuration.setup_dataset(cfg, 'train')(tr_train)
    dataset_valid = configuration.setup_dataset(cfg, 'valid')(tr_valid)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'])
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                                batch_size=cfg['batch_size'],
                                                shuffle=cfg['shuffle'])

    # Model
    model = configuration.setup_model(cfg).to(device)
    print(model)
    # TODO: checkpointing

    # Optimizer
    optim = configuration.setup_optimizer(cfg)(model.parameters())
    print(optim)

    criterion = torch.nn.MSELoss()

    # Aaaaaannnnnd, here we go!
    best_metric = 0.
    for epoch in range(num_epoch):

        train_epoch(model=model,
                       device=device,
                       optimizer=optim,
                       train_loader=train_loader,
                       criterion=criterion)

        metric = test(model=model,
                               device=device,
                               data_loader=valid_loader,
                               criterion=criterion)

        if metric > best_metric:
            best_metric = metric

    monitoring.log_experiment_csv(cfg, [best_metric])
    return best_metric

def train_epoch(model, device, train_loader, optimizer, criterion):

    model.train()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

        optimizer.zero_grad()
        data, target = data.to(device), target.to(device).float()

        output = model(data)

        loss = criterion(output.squeeze(), target.squeeze())
        loss.backward()

        optimizer.step()

def test(model, device, data_loader, criterion):

    model.eval()
    data_loss = 0

    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            
            data_loss += criterion(output.squeeze(), target.squeeze()).sum().item() # sum up batch loss
            
            targets.append(target.cpu().data.numpy())
            predictions.append(output.cpu().data.numpy())

    data_loss /= len(data_loader.dataset)
    print('\nAverage loss: {:.4f}\n'.format(data_loss))
    return data_loss

def train_skopt(cfg, n_iter, base_estimator, n_initial_points, random_state, train_function=train):

    """
    Do a Bayesian hyperparameter optimization.

    :param cfg: Configuration file.
    :param n_iter: Number of Bayesien optimization steps.
    :param base_estimator: skopt Optimization procedure.
    :param n_initial_points: Number of random search before starting the optimization.
    :param random_state: seed.
    :param train_function: The trainig procedure to optimize. The function should take a dict as input and return a metric maximize.
    :return:
    """

    import skopt
    from skopt.space import Real, Integer, Categorical

    # Helper function to help us sparse the yaml config file.
    def parse_dict(d_, prefix='', l=[]):
        """
        Find the keys in the config dict that are to be optimized.
        """
        if isinstance(d_, dict):
            for key in d_.keys():
                temp = parse_dict(d_[key], prefix + '.' + key, [])
                if temp:
                    l += temp
            return l
        else:
            try:
                x = eval(d_)
                if isinstance(x, (Real, Integer, Categorical)):
                    l.append((prefix, x))
            except:
                pass
            return l

    # Helper functions to hack in the config and change the right parameter.
    def set_key(dic, key, value):
        """
        Aux function to set the value of a key in a dict
        """
        k1 = key.split(".")
        k1 = list(filter(lambda l: len(l) > 0, k1))
        if len(k1) == 1:
            dic[k1[0]] = value
        else:
            set_key(dic[k1[0]], ".".join(k1[1:]), value)

    def generate_config(config, keys, new_values):
        new_config = copy.deepcopy(config)
        for i, key in enumerate(list(keys.keys())):
            set_key(new_config, key, new_values[i])
        return new_config

    # Sparse the parameters that we want to optimize
    skopt_args = OrderedDict(parse_dict(cfg))

    # Create the optimizer
    optimizer = skopt.Optimizer(dimensions=skopt_args.values(),
                                base_estimator=base_estimator,
                                n_initial_points=n_initial_points,
                                random_state=random_state)

    for _ in range(n_iter):

        # Do a bunch of loops.
        suggestion = optimizer.ask()
        this_cfg = generate_config(cfg, skopt_args, suggestion)
        optimizer.tell(suggestion, train_function(this_cfg)) # We minimize the L2 loss.

    # Done! Hyperparameters tuning has never been this easy.

