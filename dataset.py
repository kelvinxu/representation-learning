import os

import h5py

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
        self.f = h5py.File(os.path.join(config.data_path, 'dogs_vs_cats',
                                        'dogs_vs_cats.hdf5'))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['f']

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.f = h5py.File(os.path.join(config.data_path, 'dogs_vs_cats',
                                        'dogs_vs_cats.hdf5'))

    @property
    def num_examples(self):
        return self.stop - self.start

    def get_data(self, state=None, request=None):
        if state is not None:
            raise ValueError
        images, targets = [], []
        request = sorted([i + self.start for i in request])
        for image, shape, target in zip(self.f['images'][request],
                                        self.f['shapes'][request],
                                        self.f['labels'][request]):
            images.append(image.reshape(shape))
            targets.append([target])
        return self.filter_sources((images, targets))
