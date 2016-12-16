import cnnrandom.models as cnn_models
import os
import numpy as np
from glob import glob
from scipy import misc
from cnnrandom import BatchExtractor

DEFAULT_IN_SHAPE = (252, 324)


def retrieve_fnames(dataset_path, image_type):
    dir_names = []

    for root, subFolders, files in os.walk(dataset_path):
        for f in files:
            if f[-len(image_type):] == image_type:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)
    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(dataset_path, dir_name, '*.' + image_type)))
        fnames += dir_fnames
    return fnames


def load_imgs(fnames, out_shape):
    n_imgs = len(fnames)
    img_set = np.empty((n_imgs,) + out_shape, dtype='float32')

    for i, fname in enumerate(fnames):
        arr = misc.imread(fname, flatten=True)
        arr = misc.imresize(arr, out_shape).astype('float32')

        arr -= arr.min()
        arr /= arr.max()

        img_set[i] = arr

    return img_set


def extract_dataset(output_basename, dataset_path, output_path, image_type, model_name):
    try:
        model = eval('cnn_models.' + model_name)
    except:
        print 'problem importing model!'
        return

    fnames = retrieve_fnames(dataset_path, image_type)
    extractor = BatchExtractor(in_shape=DEFAULT_IN_SHAPE, model=model)

    print 'loading images...'
    imgs = load_imgs(fnames, DEFAULT_IN_SHAPE)

    if len(imgs) > 0:
        print 'extracting features...'
        feat_set = extractor.extract(imgs)
        feat_set.shape = feat_set.shape[0], -1

        print 'saving extracted features and corresponding list of images...'
        np.save(os.path.join(output_path, output_basename + '.npy'), feat_set)
        np.savetxt(os.path.join(output_path, output_basename + '.txt'), fnames, fmt='%s')
    else:
        print 'no images to be extracted.'

    print 'done!'


if __name__ == "__main__":
    d = os.path.abspath('../data-livdet-2015')
    DEFAULT_IMG_TYPE = 'png'
    DEFAULT_MODEL = 'fg11_ht_l3_1_description'
    dataset_dirs = {
        'test_fake': 'Testing/Digital_Persona/Fake',
        'test_live': 'Testing/Digital_Persona/Live',
        'train_fake': 'Training/Digital_Persona/Fake',
        'train_live': 'Training/Digital_Persona/Live',
    }
    for k in dataset_dirs:
        extract_dataset(k, os.path.join(d, dataset_dirs[k]), d, DEFAULT_IMG_TYPE, DEFAULT_MODEL)
