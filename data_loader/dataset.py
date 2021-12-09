import copy
import logging

import pickle

import numpy as np
import torch
# from imblearn.over_sampling import SMOTE

from data_loader.batch_graph import GGNNBatchGraph
from tqdm import tqdm

from data_loader.data_entry import DataEntry
from utils import initialize_batch
# logger = logging.getLogger(__name__)

def debug(*msg, sep=' '):
    print(sep.join((str(m) for m in msg)))

class SavedDataset:
    """
    Saved version of the dataset before splitting.
    It's very inefficient to preprocess each split separately, so we preprocess all the data
    and split it later.
    """
    def __init__(self, examples, aug_examples, edge_types, max_etype, feature_size):
        self.examples = examples
        self.aug_examples = aug_examples
        self.edge_types = edge_types
        self.max_etype = max_etype
        self.feature_size = feature_size

    def to_dataset(self, roll, splits, batch_size):
        examples = np.roll(self.examples, roll)
        train_data, valid_data, test_data = np.split(examples, splits)
        dataset = DataSet(train_data=train_data, valid_data=valid_data, test_data=test_data,
                       augmented=self.aug_examples,
                       edge_types=self.edge_types, max_etype=self.max_etype, feature_size=self.feature_size,
                       batch_size=batch_size)
        debug(f'dataset sizes: train={len(dataset.train_examples)} ({len(dataset.train_examples)-len(train_data)} augmented), valid={len(dataset.valid_examples)}, test={len(dataset.test_examples)}')
        return dataset

    @staticmethod
    def read_dataset(n_ident, g_ident, l_ident, data_src, augmented_src, use_smote=False):
        # TODO: Replace this primitive nonsense with the hardcoded edge types
        max_etype = 0
        edge_types = {}
        feature_size = 0
        def get_edge_type_number(_type):
            nonlocal max_etype
            nonlocal edge_types
            if _type not in edge_types:
                edge_types[_type] = max_etype
                max_etype += 1
            return edge_types[_type]

        def to_entry(entry):
            # graph = entry["graph"]
            # del entry["graph"]
            # entry.update(graph)
            # print(entry)
            example = DataEntry(get_edge_type_number=get_edge_type_number, num_nodes=len(entry[n_ident]), features=entry[n_ident],
                                edges=entry[g_ident], target=entry[l_ident][0][0], filename=entry["file_name"])
            return example

        examples = []
        debug('Preprocessing from %s!' % data_src)
        with open(data_src, 'rb') as fp:
            data = pickle.load(fp)
            skipped = 0
            for example in tqdm(data):
                if 'graph' not in example or 'node_features' not in example:
                    skipped += 1
                    continue
                entry = to_entry(example)
                if feature_size == 0:
                    feature_size = entry.features.size(1)
                    debug('Feature Size %d' % feature_size)
                examples.append(entry)
                del example
        debug(f'Skipped {skipped} examples')

        # if use_smote:
        #     train_features = []
        #     train_targets = []
        #     for entry in examples:
        #         train_features.append(entry.features)
        #         train_targets.append((entry.target))
        #     train_features = np.array(train_features)
        #     train_targets = np.array(train_targets)
        #     # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/762#issuecomment-705075051
        #     smote = SMOTE(random_state=0)
        #     features, targets = smote.fit_resample(train_features, train_targets)

        debug('Preprocessing from %s!' % augmented_src)
        with open(augmented_src, 'rb') as fp:
            aug_examples = pickle.load(fp)
            skipped = 0
            for aug_dataset, aug_data in aug_examples.items():
                for aug_filename, example in tqdm(list(aug_data.items()), desc=aug_dataset):
                    if 'graph' not in example:
                        skipped += 1
                        del aug_data[aug_filename]
                        continue
                    aug_data[aug_filename] = to_entry(example)
                    del example
        debug(f'Skipped {skipped} examples')

        debug(f'Partial stats {feature_size} {edge_types} {max_etype}')
        return SavedDataset(examples, aug_examples, edge_types, max_etype, feature_size)


class DataSet:
    def __init__(self, train_data, valid_data, test_data, augmented, edge_types, max_etype, feature_size, batch_size=128):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = edge_types
        self.max_etype = max_etype
        self.feature_size = feature_size
        if valid_data is None:
            valid_data = []
        if test_data is None:
            test_data = []
        if augmented is None:
            augmented = {}
        self.load_dataset(train_data, valid_data, test_data, augmented)
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def load_dataset(self, train_data, valid_data, test_data, augmented):
        for example in train_data:
            self.train_examples.append(example)
            for aug_data in augmented.values():
                if example.filename in aug_data:
                    self.train_examples.append(aug_data[example.filename])
        for example in valid_data:
            self.valid_examples.append(example)
        for example in test_data:
            self.test_examples.append(example)

    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        # print("DEBUG taken_entries:", taken_entries)
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
