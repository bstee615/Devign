#%%
import os

import torch
from torch.nn import BCELoss
import pickle
import json
import argparse

from data_loader.dataset import DataSet
from modules.model import GGNNSum
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def run_one(output_dir, model, loss_function, name, num_batches, get_fn, idx, logger_fn=print):
    all_predictions = []
    all_targets = []
    all_loss = []
    final = []
    with torch.no_grad():
        for l in range(num_batches):
            graph, targets = get_fn()
            graph.cuda(device='cuda:0')
            graph.graph = graph.graph.to('cuda:0')

            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            all_predictions.extend(
                predictions.ge(torch.ones_like(predictions) / 2)
                .detach().cpu().int().numpy().tolist())
            all_targets.extend(targets.detach().cpu().int().numpy().tolist())
            all_loss.append(batch_loss.detach().cpu().item())

            output = model.get_graph_embeddings(graph)
            final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))

    out = []
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1]})

    dst_filepath = os.path.join(output_dir, name + f'_GGNNoutput_graph-{idx}.pkl')
    logger_fn(f'Saving to {dst_filepath}')
    with open(dst_filepath, 'wb') as of:
        pickle.dump(out, of)

    out = {
        "loss": np.mean(all_loss),
        "accuracy": accuracy_score(all_targets, all_predictions),
        "precision": precision_score(all_targets, all_predictions),
        "recall": recall_score(all_targets, all_predictions),
        "f1": f1_score(all_targets, all_predictions),
    }
    json_filepath = os.path.join(output_dir, name + f'_result-{idx}.json')
    logger_fn(f'Saving intermediate results to {json_filepath}')
    with open(json_filepath, 'w') as of:
        json.dump(out, of, indent=2)

    logger_fn(f'DONE: {name} BATCHES')


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='path to GGNNSum model to be used', required=True)
    parser.add_argument('--dataset', help='path to data to be processed (i.e. processed.bin)', required=True)
    parser.add_argument('--output_dir', help='location to place data after ggnn processing', default='data/after_ggnn/chrome_debian/')
    parser.add_argument('--batch_size', help='batch size', type=int, default=128)
    args = parser.parse_args(raw_args)

    os.makedirs(args.output_dir, exist_ok=True)

    #%% data and model imports
    # (data, augmented_data) = pickle.load(open(args.dataset, 'rb'))
    saved_dataset = pickle.load(open(args.dataset, 'rb'))

    with open(os.path.join(args.model_dir, f'splits-ggnn.json')) as f:
        splits = json.load(f)
    print('Data & Models Loaded')
    print('='*83)

    for split in splits:
        i = split["idx"]
        print(f'Fold: {i}')
        model = GGNNSum(input_dim=169, output_dim=200, num_steps=6, max_edge_types=5)
        loss_function = BCELoss(reduction='sum')
        state_dict = torch.load(os.path.join(args.model_dir, f'ggnn-model-{i}.pth'))
        model.load_state_dict(state_dict, strict=False)
        model = model.to('cuda:0')
        model.eval()

        dataset = saved_dataset.to_dataset(split["roll"], split["splits"], args.batch_size)
        # current_data = np.roll(data, split["roll"])
        # train_data, valid_data, test_data = np.split(current_data, split["splits"])
        # dataset = DataSet(train_data=train_data, valid_data=valid_data, test_data=test_data,
        #                   augmented=augmented_data,
        #                   batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
        #                   l_ident=args.label_tag, inf=False)

        run_one(args.output_dir, model, loss_function, 'test', dataset.initialize_test_batches(), dataset.get_next_test_batch, i)
        run_one(args.output_dir, model, loss_function, 'valid', dataset.initialize_valid_batches(), dataset.get_next_valid_batch, i)
        run_one(args.output_dir, model, loss_function, 'train', dataset.initialize_train_batches(), dataset.get_next_train_batch, i)

if __name__ == "__main__":
    main()
