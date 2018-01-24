#!/usr/local/bin/python
'''Converts Cornell Grasping Dataset data into TFRecords data format using Example protos.
The raw data set resides in png and txt files located in the following structure:

    dataset/03/pcd0302r.png
    dataset/03/pcd0302cpos.txt
'''

import os
import errno
import traceback
import itertools
import six
import os
import glob
import numpy as np

import numpy as np
import tensorflow as tf
import re
from scipy.ndimage.filters import median_filter

# progress bars https://github.com/tqdm/tqdm
# import tqdm without enforcing it as a dependency
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras._impl.keras.utils.data_utils import _hash_file
import keras
from keras import backend as K

# TODO(ahundt) importing moviepy prevented python from exiting, uncomment lines when fixed.
# try:
#     import moviepy.editor as mpy
# except ImportError:
#     print('moviepy not available, try `pip install moviepy`. '
#           'Skipping dataset gif extraction components.')

# TODO(ahundt) these are from the google brain grasping dataset
# import grasp_geometry
# import grasp_geometry_tf
# import depth_image_encoding
# import random_crop as rcp
# import inception_preprocessing
# from grasp_median_filter import grasp_dataset_median_filter


flags.DEFINE_string('data_dir',
                    os.path.join(os.path.expanduser("~"),
                                 '.keras', 'datasets', 'cornell_grasping'),
                    """Path to dataset in TFRecord format
                    (aka Example protobufs) and feature csv files.""")
flags.DEFINE_string('grasp_dataset', 'all', 'TODO(ahundt): integrate with brainrobotdata or allow subsets to be specified')
flags.DEFINE_boolean('grasp_download', True,
                     """Download the grasp_dataset to data_dir if it is not already present.""")

FLAGS = flags.FLAGS


def mkdir_p(path):
    """Create the specified path on the filesystem like the `mkdir -p` command

    Creates one or more filesystem directory levels as needed,
    and does not return an error if the directory already exists.
    """
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def is_sequence(arg):
    """Returns true if arg is a list or another Python Sequence, and false otherwise.

        source: https://stackoverflow.com/a/17148334/99379
    """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))


class GraspDataset(object):
    """Cornell Grasping Dataset - about 5GB total size
        http:pr.cs.cornell.edu/grasping/rect_data/data.php

        Downloads to `~/.keras/datasets/cornell_grasping` by default.

        # Arguments

        data_dir: Path to dataset in TFRecord format
            (aka Example protobufs) and feature csv files.
             `~/.keras/datasets/grasping` by default.

        dataset: 'all' to load all the data.

        download: True to actually download the dataset, also see FLAGS.
    """
    def __init__(self, data_dir=None, dataset=None, download=None, verbose=0):
        if data_dir is None:
            data_dir = FLAGS.data_dir
        self.data_dir = data_dir
        if dataset is None:
            dataset = FLAGS.grasp_dataset
        self.dataset = dataset
        if download is None:
            download = FLAGS.grasp_download
        if download:
            self.download(data_dir, dataset)
        self.verbose = verbose

    def download(self, data_dir=None, dataset='all'):
        '''Cornell Grasping Dataset - about 5GB total size

        http:pr.cs.cornell.edu/grasping/rect_data/data.php

        Downloads to `~/.keras/datasets/cornell_grasping` by default.
        Includes grasp_listing.txt with all files in all datasets;
        the feature csv files which specify the dataset size,
        the features (data channels), and the number of grasps;
        and the tfrecord files which actually contain all the data.

        If `grasp_listing_hashed.txt` is present, an additional
        hashing step will will be completed to verify dataset integrity.
        `grasp_listing_hashed.txt` will be generated automatically when
        downloading with `dataset='all'`.

        # Arguments

            dataset: The name of the dataset to download, downloads all by default
                with the '' parameter, 102 will download the 102 feature dataset
                found in grasp_listing.txt.

        # Returns

           list of paths to the downloaded files

        '''
        dataset = self._update_dataset_param(dataset)
        if data_dir is None:
            if self.data_dir is None:
                data_dir = FLAGS.data_dir
            else:
                data_dir = self.data_dir
        mkdir_p(data_dir)
        print('Downloading datasets to: ', data_dir)

        url_prefix = ''
        # If a hashed version of the listing is available,
        # download the dataset and verify hashes to prevent data corruption.
        listing_hash = os.path.join(data_dir, 'grasp_listing_hash.txt')
        if os.path.isfile(listing_hash):
            files_and_hashes = np.genfromtxt(listing_hash, dtype='str', delimiter=' ')
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, file_hash=hash_str, extract=True)
                     for fpath, hash_str in tqdm(files_and_hashes)
                     if '_' + str(dataset) in fpath]
        else:
            # If a hashed version of the listing is not available,
            # simply download the dataset normally.
            listing_url = 'https://raw.githubusercontent.com/ahundt/robot-grasp-detection/master/grasp_listing.txt'
            grasp_listing_path = get_file('grasp_listing.txt', listing_url, cache_subdir=data_dir)
            grasp_files = np.genfromtxt(grasp_listing_path, dtype=str)
            files = [get_file(fpath.split('/')[-1], url_prefix + fpath, cache_subdir=data_dir, extract=True)
                     for fpath in tqdm(grasp_files)
                     if '_' + dataset in fpath]

            # If all files are downloaded, generate a hashed listing.
            if dataset is 'all' or dataset is '':
                print('Hashing all dataset files to prevent corruption...')
                hashes = []
                for i, f in enumerate(tqdm(files)):
                    hashes.append(_hash_file(f))
                file_hash_np = np.column_stack([grasp_files, hashes])
                with open(listing_hash, 'wb') as hash_file:
                    np.savetxt(hash_file, file_hash_np, fmt='%s', delimiter=' ', header='file_path sha256')
                print('Hashing complete, {} contains each url plus hash, and will be used to verify the '
                      'dataset during future calls to download().'.format(listing_hash))

        return files

    def _update_dataset_param(self, dataset):
        """Internal function to configure which subset of the datasets is being used.
        Helps to choose a reasonable default action based on previous user parameters.
        """
        if dataset is None and self.dataset is None:
            return []
        if dataset is 'all':
            dataset = ''
        if dataset is None and self.dataset is not None:
            dataset = self.dataset
        return dataset


class ImageCoder(object):
    def __init__(self):
        self._sess = tf.Session()
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
    def decode_png(self, image_data):
        return self._sess.run(self._decode_png,
                              feed_dict={self._decode_png_data: image_data})

def _process_image(filename, coder):
    # Decode the image
    with open(filename) as f:
        image_data = f.read()
    image = coder.decode_png(image_data)
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    return image_data, height, width


def _process_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes

def _int64_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def _floats_feature(v):
    if not isinstance(v, list):
        v = [v]
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))

def _bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _convert_to_example(filename, bboxes, image_buffer, height, width):
    # Build an Example proto for an example
    example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': _bytes_feature(filename),
          'image/encoded': _bytes_feature(image_buffer),
          'image/height': _int64_feature(height),
          'image/width': _int64_feature(width),
          'bboxes': _floats_feature(bboxes)}))
    return example

def main():

    gd = GraspDataset()
    if FLAGS.grasp_download:
        gd.download(dataset=FLAGS.grasp_dataset)
    train_file = os.path.join(FLAGS.data_dir, 'train-cgd')
    validation_file = os.path.join(FLAGS.data_dir, 'validation-cgd')
    print(train_file)
    print(validation_file)
    writer_train = tf.python_io.TFRecordWriter(train_file)
    writer_validation = tf.python_io.TFRecordWriter(validation_file)

    # Creating a list with all the image paths
    folders = range(1,11)
    folders = ['0'+str(i) if i<10 else '10' for i in folders]
    filenames = []
    for i in folders:
        for name in glob.glob(os.path.join(FLAGS.data_dir, i, 'pcd'+i+'*r.png')):
            filenames.append(name)

    # Shuffle the list of image paths
    np.random.shuffle(filenames)

    count = 0
    valid_img = 0
    train_img = 0

    coder = ImageCoder()
    for filename in tqdm(filenames):
        bbox = filename[:-5]+'cpos.txt'
        bboxes = _process_bboxes(bbox)
        image_buffer, height, width = _process_image(filename, coder)
        example = _convert_to_example(filename, bboxes, image_buffer, height, width)
        # Split the dataset in 80% for training and 20% for validation
        if count % 5 == 0:
            writer_validation.write(example.SerializeToString())
            valid_img +=1
        else:
            writer_train.write(example.SerializeToString())
            train_img +=1
        count +=1

    print('Done converting %d images in TFRecords with %d train images and %d validation images' % (count, train_img, valid_img))

    writer_train.close()
    writer_validation.close()


if __name__ == '__main__':
    main()
