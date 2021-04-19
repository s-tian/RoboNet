from robonet.datasets.base_dataset import BaseVideoDataset
from tensorflow.contrib.training import HParams
import os
import cv2
from robonet.datasets.util.dataset_utils import color_augment, split_train_val_test
import numpy as np
import glob
import h5py
import tensorflow as tf
import copy
import multiprocessing
import pdb
from robonet.datasets.variants.base_dataset_hdf5numeric import BaseVideoDataset
from PIL import Image


def default_loader_hparams():
    return {
    }

def preprocess_images(images, hparams):
    # Resize video
    if len(images.shape) == 5:
        images = images[:, 0]  # Number of cameras, used in RL environments

    images = np.stack([np.asarray(Image.fromarray(im)) for im in images], axis=0)
    assert images.dtype == np.uint8, 'image need to be uint8!'
    images = images[:, None]  # add dimension for ncam

    if hparams.resize_image:
        resized_images = np.zeros([images.shape[0], 1, hparams.img_size[0], hparams.img_size[1], 3], dtype=np.uint8)
        for t in range(images.shape[0]):
            resized_images[t] = cv2.resize(images[t].squeeze(), (hparams.img_size[1], hparams.img_size[0]), interpolation=cv2.INTER_CUBIC)[None]
        images = resized_images
    else:
        assert np.all(hparams.img_size == images.shape[2:3])
    return images


def _load_data(inputs):

    f_name, hparams = inputs

    load_T = hparams.load_T

    with h5py.File(f_name, 'r') as F:
        traj_per_file = F['traj_per_file'].value

        index = np.random.randint(traj_per_file)

        ex_index = index % traj_per_file  # get the index
        key = 'traj{}'.format(ex_index)

        # Fetch data into a dict
        data_dict = {'images': preprocess_images(F[key + '/images'].value, hparams)}

        for name in F[key].keys():
            if name in ['states', 'actions', 'pad_mask']:  # this is unnecessary
                data_dict[name] = F[key + '/' + name].value.astype(np.float32)
            # print('datashape {} {}'.format(name, data_dict[name].shape))

        assert data_dict['images'].shape[0] == data_dict['states'].shape[0], "imageshape {} states shape {}".format(data_dict['images'].shape, data_dict['states'].shape)
        assert data_dict['images'].shape[0]-1 == data_dict['actions'].shape[0]

        if hparams.random_shifts:
            delta = data_dict['actions'].shape[0] - load_T + 2
            shift = np.random.randint(0, delta)
        else:
            shift = 0
        for name, value in data_dict.items():
            if name == 'actions':
                data_dict[name] = value[shift:shift + load_T - 1]
            else:
                data_dict[name] = value[shift:shift + load_T]

    return data_dict['images'], data_dict['actions'], data_dict['states']


def _get_max_len(file):
    with h5py.File(file, 'r') as F:
        return F['traj0/pad_mask/'].shape[0]


class HDF5NumericRoboNetDataset(BaseVideoDataset):
    def __init__(self, batch_size, data_dir, hparams=dict()):
        self.data_dir = data_dir
        super().__init__(batch_size, data_dir, hparams)

    def _init_dataset(self):

        # set output format
        output_format = [tf.uint8, tf.float32, tf.float32]
        output_format = tuple(output_format)

        # smallest max step length of all dataset sources

        n_workers = min(self._batch_size, multiprocessing.cpu_count())
        if self._hparams.pool_workers:
            n_workers = min(self._hparams.pool_workers, multiprocessing.cpu_count())
        self._pool = multiprocessing.Pool(n_workers)

        n_train_ex = 0
        mode_sources = []
        for mode in self.modes:
            path = os.path.join(self.data_dir, os.path.join('hdf5', mode) + '/*')
            files = sorted(glob.glob(path))
            print('{} files {}'.format(mode, len(files)))
            assert len(files) > 0, 'no {} files found!'.format(mode)
            mode_sources.append(files)

        self.load_T = self._hparams.load_T

        self._place_holder_dict = self._get_placeholders()
        self._mode_generators = {}

        for name, source_files in zip(self.modes, mode_sources):
            if source_files:
                if name == 'train':
                    n_train_ex = len(source_files)

                gen_func = self._wrap_generator(source_files)
                if name == self.primary_mode:
                    dataset = tf.data.Dataset.from_generator(gen_func, output_format)
                    dataset = dataset.map(self._get_dict)
                    dataset = dataset.prefetch(self._hparams.buffer_size)
                    self._data_loader_dict = dataset.make_one_shot_iterator().get_next()
                else:
                    self._mode_generators[name] = gen_func()

        return n_train_ex


    def _get(self, key, mode):
        if mode == self.primary_mode:
            return self._data_loader_dict[key]

        if key == 'images':
            return self._img_tensor

        return self._place_holder_dict[key]

    def __contains__(self, item):
        return item in self._place_holder_dict

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'buffer_size': 100,  # examples to prefetch
            'all_modes_max_workers': True,  # use multi-threaded workers regardless of the mode
            'pool_workers': 0,  # number of workers for pool (if 0 uses batch_size workers)
            'color_augmentation': 0.0,  # std of color augmentation (set to 0 for no augmentations)
            'source_selection_probabilities':None,
            'splits':[None],
            'load_T':None,
            'load_random_cam': True,
            'random_shifts': False, # shift sequence randomly for data augmentation
            'resize_image': False,
            'target_adim': 3,
            'target_sdim': 3,
            'img_size': (64, 64),
            'load_annotations': False,
            'zero_if_missing_annotation': False,
        }
        for k, v in default_loader_hparams().items():
            default_dict[k] = v

        return HParams(**default_dict)

    def _wrap_generator(self, source_files):
        return lambda: self._hdf5_generator(source_files)

    def _hdf5_generator(self, source_files):
        while True:

            file_names, file_metadata = [], []
            while len(file_names) < self._batch_size:
                file_names.append(source_files[np.random.randint(0, len(source_files))])

            batch_jobs = [(fn, self._hparams) for fn in file_names]
            batches = self._pool.map_async(_load_data, batch_jobs).get()


            ret_vals = []
            for i, b in enumerate(batches):
                if i == 0:
                    for value in b:
                        ret_vals.append([value[None]])
                else:
                    for v_i, value in enumerate(b):
                        ret_vals[v_i].append(value[None])
                        # print(value.shape)
            ret_vals = [np.concatenate(v) for v in ret_vals]

            yield tuple(ret_vals)

    def _get_dict(self, *args):
        images, actions, states = args

        out_dict = {}
        height, width = self._hparams.img_size

        ncam = 1
        shaped_images = tf.reshape(images, [self.batch_size, self.load_T, ncam, height, width, 3])

        out_dict['images'] = tf.cast(shaped_images, tf.float32) / 255.0
        if self._hparams.color_augmentation:
            out_dict['images'] = color_augment(out_dict['images'], self._hparams.color_augmentation)

        out_dict['actions'] = tf.reshape(actions, [self.batch_size, self.load_T - 1, self._hparams.target_adim])
        out_dict['states'] = tf.reshape(states, [self.batch_size, self.load_T, self._hparams.target_sdim])

        return out_dict

    def _get_placeholders(self):
        height, width = self._hparams.img_size
        ncam = 1

        pl_dict = {}

        img_pl = tf.placeholder(tf.uint8, shape=[self.batch_size, self.load_T, ncam, height, width, 3])
        self._img_tensor = tf.cast(img_pl, tf.float32) / 255.0
        if self._hparams.color_augmentation:
            self._img_tensor = color_augment(self._img_tensor, self._hparams.color_augmentation)

        pl_dict['images'] = img_pl
        pl_dict['actions'] = tf.placeholder(tf.float32, shape=[self.batch_size, self.load_T - 1,
                                                               self._hparams.target_adim])
        pl_dict['states'] = tf.placeholder(tf.float32,
                                           [self.batch_size, self.load_T, self._hparams.target_sdim])
        # for key in pl_dict.keys():
        #     print('pl dict {} {}'.format(key, pl_dict[key].shape))

        return pl_dict

    def build_feed_dict(self, mode):
        fetch = {}
        if mode == self.primary_mode:
            # set placeholders to null
            images = np.zeros(self._place_holder_dict['images'].get_shape().as_list(), dtype=np.uint8)
            actions = np.zeros(self._place_holder_dict['actions'].get_shape().as_list(), dtype=np.float32)
            states = np.zeros(self._place_holder_dict['states'].get_shape().as_list(), dtype=np.float32)
        else:
            args = next(self._mode_generators[mode])
            images, actions, states = args

        fetch[self._place_holder_dict['images']] = images
        fetch[self._place_holder_dict['actions']] = actions
        fetch[self._place_holder_dict['states']] = states

        return fetch

def _timing_test(N, loader):
    import time
    import random

    mode_tensors = {}
    for m in loader.modes:
        mode_tensors[m] = [loader[x, m] for x in ['images', 'states', 'actions']]
    s = tf.Session()

    timings = []
    for m in loader.modes:
        for i in range(N):

            start = time.time()
            s.run(mode_tensors[m], feed_dict=loader.build_feed_dict(m))
            run_time = time.time() - start
            if m == 'train':
                timings.append(run_time)
            print('run {}, mode {} took {} seconds'.format(i, m, run_time))

    if timings:
        print('train runs took on average {} seconds'.format(sum(timings) / len(timings)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="calculates or loads meta_data frame")
    parser.add_argument('path', help='path to files containing hdf5 dataset')
    parser.add_argument('--robots', type=str, nargs='+', default=None,
                        help='will construct a dataset with batches split across given robots')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for test loader (should be even for non-time test demo to work)')
    parser.add_argument('--mode', type=str, default='train', help='mode to grab data from')
    parser.add_argument('--time_test', type=int, default=0, help='if value provided will run N timing tests')
    parser.add_argument('--load_steps', type=int, default=0, help='if value is provided will load <load_steps> steps')
    args = parser.parse_args()

    hparams = {'load_T': args.load_steps, 'resize_image':True, 'random_shifts':False,'target_adim': 2, 'target_sdim': 5}

    loader = HDF5NumericRoboNetDataset(args.batch_size, args.path, hparams=hparams)

    if args.time_test:
        _timing_test(args.time_test, loader)
        exit(0)

    tensors = [loader[x, args.mode] for x in ['images', 'states', 'actions']]
    s = tf.Session()

    out_tensors = s.run(tensors, feed_dict=loader.build_feed_dict(args.mode))

    import imageio

    writer = imageio.get_writer('test_frames.gif')
    for t in range(out_tensors[0].shape[1]):
        writer.append_data((np.concatenate([b for b in out_tensors[0][:, t, 0]], axis=-2) * 255).astype(np.uint8))
    writer.close()
    import pdb;

    pdb.set_trace()
    print('loaded tensors!')
