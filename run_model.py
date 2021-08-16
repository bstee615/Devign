#%%
import torch
from torch.nn import BCELoss
import pickle
import json
import argparse
from modules.model import GGNNSum
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='path to GGNNSum model to be used', default='models/FULL/GGNNSumModel-model.bin')
parser.add_argument('--dataset', help='path to data to be processed (i.e. processed.bin)', default='../ReVeal/data/full_experiment_real_data_processed/chrome_debian/processed.bin')
parser.add_argument('--output_dir', help='location to place data after ggnn processing', default='after_ggnn/chrome_debian/')
parser.add_argument('--name', help='name of folder to save data in (to differentiate sets)', default='testRun')

args = parser.parse_args()

#%% data and model imports
dataset = pickle.load(open(args.dataset, 'rb'))

state_dict = torch.load(args.model)

_model = GGNNSum(input_dim=169, output_dim=200, num_steps=6, max_edge_types=5)
loss_function = BCELoss(reduction='sum')

_model.load_state_dict(state_dict, strict=False)
_model = _model.to('cuda:0')
_model.eval()
print('Data & Models Loaded')
print('='*83)

#%% get ggnn data from test batches
final = []
all_predictions = []
all_targets = []
all_loss = []
for l in range(len(dataset.test_batches)+1):
    if dataset.inf:
        graph, targets, file_names = dataset.get_next_test_batch() 
    else:
        graph, targets = dataset.get_next_test_batch()
    graph.cuda(device='cuda:0')
    graph.graph = graph.graph.to('cuda:0')
    output = _model.get_graph_embeddings(graph)
    if dataset.inf:
        final += list(zip(output.tolist(), [int(i) for i in targets.tolist()], file_names))
    else:
        final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))
    targets = targets.cuda()
    predictions = _model(graph, cuda=True)
    batch_loss = loss_function(predictions, targets)
    all_predictions.extend(
        predictions.ge(torch.ones_like(predictions) / 2)
        .detach().cpu().int().numpy().tolist())
    all_targets.extend(targets.detach().cpu().int().numpy().tolist())
    all_loss.append(batch_loss.detach().cpu().item())

out = {
    "loss": np.mean(all_loss),
    "accuracy": accuracy_score(all_targets, all_predictions),
    "precision": precision_score(all_targets, all_predictions),
    "recall": recall_score(all_targets, all_predictions),
    "f1": f1_score(all_targets, all_predictions),
}
with open(args.output_dir + args.name + '/test_result.json', 'w') as of:
    json.dump(out, of, indent=2)

out = []
if dataset.inf:
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1], 'file_name':f[2]})
else:
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1]})

with open(args.output_dir + args.name + '/test_GGNNinput_graph.json', 'w') as of:
    json.dump(out, of, indent=2)
    of.close()

print('DONE: TEST BATCHES')

#%% get ggnn data from valid batches
final = []
all_predictions = []
all_targets = []
all_loss = []
for l in range(len(dataset.valid_batches)+1):
    if dataset.inf:
        graph, targets, file_names = dataset.get_next_valid_batch() 
    else:
        graph, targets = dataset.get_next_valid_batch()
    graph.cuda(device='cuda:0')
    graph.graph = graph.graph.to('cuda:0')

    output = _model.get_graph_embeddings(graph)
    if dataset.inf:
        final += list(zip(output.tolist(), [int(i) for i in targets.tolist()], file_names))
    else:
        final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))
    targets = targets.cuda()
    predictions = _model(graph, cuda=True)
    batch_loss = loss_function(predictions, targets)
    all_predictions.extend(
        predictions.ge(torch.ones_like(predictions) / 2)
        .detach().cpu().int().numpy().tolist())
    all_targets.extend(targets.detach().cpu().int().numpy().tolist())
    all_loss.append(batch_loss.detach().cpu().item())

out = {
    "loss": np.mean(all_loss),
    "accuracy": accuracy_score(all_targets, all_predictions),
    "precision": precision_score(all_targets, all_predictions),
    "recall": recall_score(all_targets, all_predictions),
    "f1": f1_score(all_targets, all_predictions),
}
with open(args.output_dir + args.name + '/valid_result.json', 'w') as of:
    json.dump(out, of, indent=2)

out = []
if dataset.inf:
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1], 'file_name':f[2]})
else:
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1]})

with open(args.output_dir + args.name + '/valid_GGNNinput_graph.json', 'w') as of:
    json.dump(out, of, indent=2)
    of.close()

print('DONE: VALID BATCHES')

#%% get ggnn data from train batches
final = []
all_predictions = []
all_targets = []
all_loss = []
for l in range(len(dataset.train_batches)+1):
    if dataset.inf:
        graph, targets, file_names = dataset.get_next_train_batch() 
    else:
        graph, targets = dataset.get_next_train_batch()
    graph.cuda(device='cuda:0')
    graph.graph = graph.graph.to('cuda:0')

    output = _model.get_graph_embeddings(graph)
    if dataset.inf:
        final += list(zip(output.tolist(), [int(i) for i in targets.tolist()], file_names))
    else:
        final += list(zip(output.tolist(), [int(i) for i in targets.tolist()]))
    targets = targets.cuda()
    predictions = _model(graph, cuda=True)
    batch_loss = loss_function(predictions, targets)
    all_predictions.extend(
        predictions.ge(torch.ones_like(predictions) / 2)
        .detach().cpu().int().numpy().tolist())
    all_targets.extend(targets.detach().cpu().int().numpy().tolist())
    all_loss.append(batch_loss.detach().cpu().item())

out = {
    "loss": np.mean(all_loss),
    "accuracy": accuracy_score(all_targets, all_predictions),
    "precision": precision_score(all_targets, all_predictions),
    "recall": recall_score(all_targets, all_predictions),
    "f1": f1_score(all_targets, all_predictions),
}
with open(args.output_dir + args.name + '/train_result.json', 'w') as of:
    json.dump(out, of, indent=2)

out = []
if dataset.inf:
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1], 'file_name':f[2]})
else:
    for f in final:
        out.append({'graph_feature':f[0], 'target':f[1]})

with open(args.output_dir + args.name + '/train_GGNNinput_graph.json', 'w') as of:
    json.dump(out, of, indent=2)
    of.close()

print('DONE: TRAIN BATCHES')

#%%
print('='*83)
print('COMPLETED')
print('='*83)
