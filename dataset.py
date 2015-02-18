import os

import numpy

from fuel import config
from fuel.datasets import InMemoryDataset


@InMemoryDataset.lazy_properties('features', 'targets')
class DogsVsCats(InMemoryDataset):
    provides_sources = ('features', 'targets')

    def __init__(self, which_set):
        if which_set == 'train':
            self.start = 0
            self.stop = 20000
        elif which_set == 'valid':
            self.start = 20000
            self.stop = 22500
        elif which_set == 'test':
            self.start = 22500
            self.stop = 25000
        else:
            raise ValueError

    def load(self):
        path = os.path.join(config.data_path, 'dogs_vs_cats')
        self.features = numpy.load(os.path.join(path, 'features.npy'))
        self.targets = numpy.load(os.path.join(path, 'targets.npy'))

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError
        return self.filter_sources((self.features[request],
                                    self.targets[request]))
