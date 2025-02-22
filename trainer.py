import copy
import datetime

import logging

from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# logger = logging.getLogger(__name__)

def debug(*msg, sep=' '):
    print(sep.join((str(m) for m in msg)))

def evaluate_loss(model, loss_function, num_batches, data_iter, return_all=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            graph.cuda(device='cuda:0')
            graph.graph = graph.graph.to('cuda:0')
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            #print('predictions.ndim:', predictions.ndim)
            if predictions.ndim == 2:
                all_predictions.extend(
                    torch.argmax(predictions, axis=-1)
                    .detach().cpu().int().numpy().tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones_like(predictions) / 2)
                    .detach().cpu().int().numpy().tolist())
            all_targets.extend(targets.detach().cpu().int().numpy().tolist())
        model.train()
        if return_all:
            return np.mean(_loss).item(),\
                   accuracy_score(all_targets, all_predictions) * 100,\
                   precision_score(all_targets, all_predictions) * 100, \
                   recall_score(all_targets, all_predictions) * 100,\
                   f1_score(all_targets, all_predictions) * 100
        else:
            return np.mean(_loss).item(), f1_score(all_targets, all_predictions) * 100


def evaluate_metrics(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            graph.cuda(device='cuda:0')
            graph.graph = graph.graph.to('cuda:0')
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            #print('predictions.ndim:', predictions.ndim)
            if predictions.ndim == 2:
                all_predictions.extend(
                    torch.argmax(predictions, axis=-1)
                    .detach().cpu().int().numpy().tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones_like(predictions) / 2)
                    .detach().cpu().int().numpy().tolist())
            all_targets.extend(targets.detach().cpu().int().numpy().tolist())
        model.train()
        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, max_steps, dev_every, loss_function, optimizer, save_path, output_file, log_every=50, max_patience=5, ray=False):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    try:
        dataset.initialize_valid_batch()
        for step_count in range(max_steps):
            model.train()
            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()
            graph.cuda(device='cuda:0')
            graph.graph = graph.graph.to('cuda:0')
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            if log_every is not None and (step_count % log_every == log_every - 1):
                debug('Step %d\t\tTrain Loss %10.3f' % (step_count, batch_loss.detach().cpu().item()))
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
            if step_count % dev_every == (dev_every - 1):
                # valid_loss, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_valid_batch(),
                #                                      dataset.get_next_valid_batch)
                valid_loss, valid_acc, valid_prec, valid_rec, valid_f1 = evaluate_loss(model, loss_function, dataset.initialize_valid_batch(),
                                                     dataset.get_next_valid_batch, return_all=True)
                if ray:
                    from ray import tune
                    tune.report(valid_loss=valid_loss,
                                valid_acc=valid_acc, valid_prec=valid_prec,
                                valid_rec=valid_rec, valid_f1=valid_f1)
                if valid_f1 > best_f1:
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(save_path, 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                else:
                    patience_counter += 1
                debug(datetime.datetime.now(), 'Step %d\t\tTrain Loss %10.3f\tValid Loss%10.3f\tValid f1: %5.2f\tPatience %d' % (
                    step_count, np.mean(train_losses).item(), valid_loss, valid_f1, patience_counter))
                debug('=' * 100)
                train_losses = []
                if patience_counter == max_patience:
                    break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(save_path, 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()
    acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)
    debug(datetime.datetime.now(), '%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    debug('=' * 100)

    print('Test:', acc, pr, rc, f1, flush=True, file=output_file)
