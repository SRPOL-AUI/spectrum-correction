"""Sequences for keras"""

import h5py
import numpy as np

from tensorflow.python.keras.utils import Sequence


class HDF5Batch(Sequence):
    """Sequence reading mini-batches from a HDF5 file.

    Args:
        batch_size (int): size of the mini-batch
        file (File): HDF5 file
        features_dataset (str): name of the dataset containing features
        labels_dataset (str): name of the dataset containing labels
    """

    def __init__(self, batch_size, file, features_dataset, labels_dataset) -> None:
        super().__init__()

        self._batch_size = batch_size

        h5 = file if isinstance(file, h5py.File) else h5py.File(file, 'r')
        self._features = h5[features_dataset]
        self._labels = h5[labels_dataset]

        if len(self._features) != len(self._labels):
            raise ValueError('Datasets must have the same length.')

    def __len__(self):
        return len(self._labels) // self._batch_size

    def __getitem__(self, index):
        return (
            self._features[index * self._batch_size : (index + 1) * self._batch_size],
            self._labels[index * self._batch_size : (index + 1) * self._batch_size],
        )


class Sparse2OneHot(Sequence):
    """Sequence that converts sparse labels to one-hot vectors.

    Args:
        sequence (Sequence): another sequence
        depth (int): depth of the one-hot vector (number of classes)
    """

    def __init__(self, sequence, depth):
        super().__init__()
        self.sequence = sequence
        self.depth = depth

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        examples, labels = self.sequence[index]
        labels = np.eye(self.depth)[labels]
        return examples, labels


class Mixup(Sequence):
    """Sequence that performs mixup augmentation.

    Args:
        sequence: another sequence
        alpha: alpha parameter for the augmentation
        use_exp: perform element wise exp(.) before mixing then log(.)
    """

    def __init__(self, sequence, alpha, use_exp):
        super().__init__()
        self.sequence = sequence
        self.alpha = alpha
        self.use_exp = use_exp

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        other_index = np.random.randint(len(self.sequence))
        examples, labels = self.sequence[index]
        other_examples, other_labels = self.sequence[other_index]

        assert np.array_equal(examples.shape, other_examples.shape)
        a = np.random.beta(self.alpha, self.alpha, examples.shape[0])

        if self.use_exp:
            examples = np.exp(examples)
            other_examples = np.exp(other_examples)

        format = 'i...,i...->i...'
        labels = np.einsum(format, a, labels) + np.einsum(format, 1 - a, other_labels)
        examples = np.einsum(format, a, examples) + np.einsum(format, 1 - a, other_examples)

        if self.use_exp:
            examples = np.log(examples)

        return examples, labels
