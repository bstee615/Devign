import numpy as np
import logging

from data_loader import n_identifier, g_identifier, l_identifier


logger = logging.getLogger(__name__)


def load_default_identifiers(n, g, l):
    if n is None:
        n = n_identifier
    if g is None:
        g = g_identifier
    if l is None:
        l = l_identifier
    return n, g, l


def initialize_batch(entries, batch_size, shuffle=False):
    total = len(entries)
    indices = np.arange(0, total - 1, 1)
    if shuffle:
        np.random.shuffle(indices)
    batch_indices = []
    start = 0
    end = len(indices)
    curr = start
    while curr < end:
        c_end = curr + batch_size
        if c_end > end:
            c_end = end
        batch_indices.append(indices[curr:c_end])
        curr = c_end
    return batch_indices[::-1]


def tally_param(model):
    total = 0
    for param in model.parameters():
        total += param.data.nelement()
    return total


def debug(*msg, sep=' '):
    logger.info(sep.join((str(m) for m in msg)))
