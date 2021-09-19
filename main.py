import argparse
import os
import pickle
import random
import sys

import numpy as np
import sklearn.model_selection
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
from trainer import train
from utils import tally_param, debug
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--model_dir', type=str, required=True, help='Output file for the best model')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=169)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)
    parser.add_argument('--patience', type=int, help='Patience for early stopping', default=50)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--preprocess_only', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dir_tmp = args.input_dir
    if input_dir_tmp.endswith('/'):
        input_dir_tmp = input_dir_tmp[:-1]

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    input_dir = args.input_dir
    logfile_name = f'devign-{args.model_type}'
    if args.preprocess_only:
        logfile_name += '-preprocess_only'
    logfile_name += '.log'
    logfile_path = os.path.join(model_dir, logfile_name)
    if os.path.exists(logfile_path):
        os.unlink(logfile_path)
    logger.addHandler(logging.FileHandler(logfile_path))

    if args.feature_size > args.graph_embed_size:
        logger.error('Graph Embed dimension should be at least equal to the feature dimension.\n'
                     'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    processed_data_path = os.path.join(input_dir, 'processed.bin')

    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.pkl'),
                          valid_src=os.path.join(input_dir, 'valid_GGNNinput.pkl'),
                          test_src=os.path.join(input_dir, 'test_GGNNinput.pkl'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag, inf=False)
        with open(processed_data_path, 'wb') as file:
            pickle.dump(dataset, file)
    logger.info(f'Feature size: {dataset.feature_size}')
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    debug(f'dataset sizes: {len(dataset.train_examples)}, {len(dataset.valid_examples)}, {len(dataset.test_examples)}')

    if args.preprocess_only:
        debug('Done preprocessing, exiting.')
        exit(0)

    logger.info(f"CUDA: {torch.cuda.is_available()}, {torch.version.cuda}")
    assert torch.cuda.is_available()

    output_file_name = os.path.join(model_dir, f'devign-{args.model_type}-results.tsv')

    with open(output_file_name, 'w') as output_file:
        model_filename = os.path.join(model_dir, f'{args.model_type}-model.pth')
        if args.model_type == 'ggnn':
            model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
        else:
            model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                                num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
        debug('Total Parameters : %d' % tally_param(model))
        debug('#' * 100)
        model.cuda()
        loss_function = BCELoss(reduction='sum')
        optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        train(model=model, dataset=dataset, max_steps=1000000, log_every=None, dev_every=128,
              loss_function=loss_function, optimizer=optim,
              save_path=model_filename, output_file=output_file, max_patience=args.patience)
