import numpy
from os import listdir
from os.path import isfile, join

import h5py
import numpy
from scipy import misc

rng = numpy.random.RandomState(123522)

path = '/data/lisatmp3/xukelvin/'

if __name__ == "__main__":
    files = [f for f in listdir(join('train'))
             if isfile(join('train', f))]

    # Shuffle examples around
    rng.shuffle(files)

    # Create HDF5 file
    # train
    print "Processing Train"
    train_f = h5py.File(path+'dogs_vs_cats_train.hdf5', 'w')
    dt = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features = train_f.create_dataset('images', (20000,), dtype=dt)
    shapes = train_f.create_dataset('shapes', (20000, 3), dtype='uint16')
    targets = train_f.create_dataset('labels', (20000,), dtype='uint8')

    for i in xrange(0,20000):
        f = files[i]
        image = misc.imread(join('train', f))
        target = 0 if 'cat' in f else 1
        features[i] = image.flatten()
        targets[i] = target
        shapes[i] = image.shape
        print '{:.0%}\r'.format(i / 20000.),

    # val
    print "Processing Validation"
    val_f = h5py.File(path+'dogs_vs_cats_val.hdf5', 'w')
    dt = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features = val_f.create_dataset('images', (2500,), dtype=dt)
    shapes = val_f.create_dataset('shapes', (2500, 3), dtype='uint16')
    targets = val_f.create_dataset('labels', (2500,), dtype='uint8')

    for i in xrange(20000,22500):
        f = files[i]
        image = misc.imread(join('train', f))
        target = 0 if 'cat' in f else 1
        features[i-20000] = image.flatten()
        targets[i-20000] = target
        shapes[i-20000] = image.shape
        print '{:.0%}\r'.format(i / 2500.),


    # test
    print "Processing Test"
    test_f = h5py.File(path+'dogs_vs_cats_test.hdf5', 'w')
    dt = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features = test_f.create_dataset('images', (2500,), dtype=dt)
    shapes = test_f.create_dataset('shapes', (2500, 3), dtype='uint16')
    targets = test_f.create_dataset('labels', (2500,), dtype='uint8')
    for i in xrange(22500,25000):
        f = files[i]
        image = misc.imread(join('train', f))
        target = 0 if 'cat' in f else 1
        features[i-22500] = image.flatten()
        targets[i-22500] = target
        shapes[i-22500] = image.shape
        print '{:.0%}\r'.format(i / 2500.),
