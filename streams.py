from collections import OrderedDict

import theano
import numpy
from scipy.misc import imresize

from fuel.streams import DataStreamWrapper


class RandomPatch(DataStreamWrapper):
    def __init__(self, data_stream, patch_size, source='features'):
        super(RandomPatch, self).__init__(data_stream)
        self.patch_size = patch_size
        self.patch_source = source

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        patch_height, patch_width = self.patch_size
        data = OrderedDict(zip(self.sources, next(self.child_epoch_iterator)))
        new_data = OrderedDict(zip(self.sources, [[] for _ in self.sources]))
        for i, image in enumerate(data[self.patch_source]):
            _, image_height, image_width = image.shape
            if image_height < patch_height or image_width < patch_width:
                continue
            x = image_width - patch_width
            y = image_height - patch_height
            if x:
                x = numpy.random.randint(x)
            if y:
                y = numpy.random.randint(y)
            patch = image[:, y:y + patch_width, x:x + patch_height]
            new_data[self.patch_source].append(patch)
            for source in self.sources:
                if source != self.patch_source:
                    new_data[source].append(data[source][i])
        new_data = tuple(numpy.asarray(source_data)
                         for source_data in new_data.values())
        return new_data


class Resize(DataStreamWrapper):
    def __init__(self, data_stream, size, source='features'):
        super(Resize, self).__init__(data_stream)
        self.size = size
        self.source = source

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        height, width = self.size
        data = OrderedDict(zip(self.sources, next(self.child_epoch_iterator)))
        new_data = OrderedDict(zip(self.sources, [[] for _ in self.sources]))
        for i, image in enumerate(data[self.source]):
            new_data[self.source].append(
                imresize(image, self.size).transpose((2, 0, 1))
                .astype(theano.config.floatX) / 255)
            for source in self.sources:
                if source != self.source:
                    new_data[source].append(data[source][i])
        return tuple(numpy.asarray(source_data)
                     for source_data in new_data.values())
