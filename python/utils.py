"""Small utilities shared by the training and reporting code."""

import numpy as np


def batches(indices, batch_size, shuffle=True):
    order = np.copy(indices)
    if shuffle:
        np.random.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def get_idx(scenario_index, shape_index, sample_size_index):
    """Return the bandwidth index used by the main paper table."""
    bandwidth_indices = [
        [[0, 0, 0], [3, 1, 1]],
        [[2, 1, 0], [4, 1, 0]],
        [[1, 1, 0], [2, 0, 0]],
    ]
    return bandwidth_indices[scenario_index][shape_index][sample_size_index]
