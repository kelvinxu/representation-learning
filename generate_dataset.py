import numpy
from os import listdir
from os.path import isfile, join
from scipy import misc

rng = numpy.random.RandomState(123522)


if __name__ == "__main__":
    files = [f for f in listdir(join('train'))
             if isfile(join('train', f))]

    # Shuffle examples around
    rng.shuffle(files)
    X, y = [], []
    for i, f in enumerate(files):
        image = misc.imread(join('train', f))
        X.append(image)
        target = 0 if 'cat' in f else 1
        y.append([target])
        print '{:.0%}\r'.format(i / 25000.),
    numpy.save('features.npy', numpy.asarray(X))
    numpy.save('targets.npy', numpy.asarray(y))
