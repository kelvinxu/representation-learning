from collections import OrderedDict

import numpy

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
            image_height, image_width, _ = image.shape
            if image_height < patch_height or image_width < patch_width:
                continue
            x = numpy.random.randint(image_width - patch_width)
            y = numpy.random.randint(image_height - patch_height)
            patch = image[x:x + patch_width, y:y + patch_height]
            new_data[self.patch_source].append(patch)
            for source in self.sources:
                if source != self.patch_source:
                    new_data[source].append(data[source][i])
        return tuple(numpy.asarray(source_data)
                     for source_data in new_data.values())
