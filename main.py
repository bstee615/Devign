import argparse
import glob
import json
import os
import pickle
import random
import sys

import numpy as np
import sklearn.model_selection
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet, SavedDataset
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

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--model_dir', type=str, default=None, help='(DEPRECATED) Directory to store the model')
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
    args = parser.parse_args(raw_args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = os.path.join(args.input_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    input_dir = os.path.join(args.input_dir, 'ggnn_input')
    logfile_name = f'devign-{args.model_type}'
    if args.preprocess_only:
        logfile_name += '-preprocess_only'
    logfile_name += '.log'
    logfile_path = os.path.join(model_dir, logfile_name)
    if os.path.exists(logfile_path):
        os.unlink(logfile_path)
    logger.addHandler(logging.FileHandler(logfile_path))

    if args.model_dir is not None:
        debug(f'--model_dir set to {args.model_dir} but is DEPRECATED. Will not be used.')

    if args.feature_size > args.graph_embed_size:
        logger.error('Graph Embed dimension should be at least equal to the feature dimension.\n'
                     'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    processed_data_path = os.path.join(input_dir, 'processed.bin')

    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        saved_dataset = pickle.load(open(processed_data_path, 'rb'))
    else:
        saved_dataset = SavedDataset.read_dataset(args.node_tag, args.graph_tag, args.label_tag, os.path.join(input_dir, 'GGNNinput.pkl'), os.path.join(input_dir, 'augmented_GGNNinput.pkl'))
        with open(processed_data_path, 'wb') as file:
            pickle.dump(saved_dataset, file)

    if args.preprocess_only:
        debug('Done preprocessing, exiting.')
        exit(0)

    logger.info(f"CUDA: {torch.cuda.is_available()}, {torch.version.cuda}")
    assert torch.cuda.is_available()

    output_file_name = os.path.join(model_dir, f'devign-{args.model_type}-results.tsv')

    all_splits = []
    with open(output_file_name, 'w') as output_file:
        n_folds = 5
        for i in range(0, n_folds):
            logger.info(f'Fold: {i}')
            roll = len(saved_dataset.examples) // n_folds * i
            splits = (int(.7*len(saved_dataset.examples)), int(.8*len(saved_dataset.examples)))
            all_splits.append({
                "idx": i,
                "roll": roll,
                "splits": splits,
            })
            dataset = saved_dataset.to_dataset(roll, splits, args.batch_size)
            with open(os.path.join(model_dir, f'splits-{args.model_type}.json'), 'w') as f:
                json.dump(all_splits, f)
            logger.info(f'Feature size: {dataset.feature_size}')
            assert args.feature_size == dataset.feature_size, \
                'Dataset contains different feature vector than argument feature size. ' \
                'Either change the feature vector size in argument, or provide different dataset.'
            model_filename = os.path.join(model_dir, f'{args.model_type}-model-{i}.pth')
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

if __name__ == "__main__":
    main()
