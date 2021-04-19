import os
import tensorflow as tf
import pdb
from tensorflow.contrib.training import HParams
import glob
import copy
import random
import numpy as np


class BaseVideoDataset(object):
    def __init__(self, batch_size, dataset_files_or_metadata, hparams=dict()):
        assert isinstance(batch_size, int), "batch_size must be an integer"
        self._batch_size = batch_size

        self.meta_data = dataset_files_or_metadata

        # initialize hparams and store metadata_frame
        self._hparams = self._get_default_hparams().override_from_dict(hparams)

        #initialize dataset
        self._num_ex_per_epoch = self._init_dataset()
        print('loaded {} train files'.format(self._num_ex_per_epoch))

    def _init_dataset(self):
        return 0

    def _get(self, key, mode):
        raise NotImplementedError

    @staticmethod
    def _get_default_hparams():
        default_dict = {
            'RNG': 11381294392481135266,
            'use_random_train_seed': False
        }
        return HParams(**default_dict)
    
    def get(self, key, mode='train'):
        if mode not in self.modes:
            raise ValueError('Mode {} not valid! Dataset has following modes: {}'.format(mode, self.modes))
        return self._get(key, mode)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                raise KeyError('Index should be in format: [Key, Mode] or [Key] (assumes default train mode)')
            key, mode = item
            return self.get(key, mode)

        return self.get(item)
    
    def __contains__(self, item):
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def hparams(self):
        return self._hparams.values()

    @property
    def num_examples_per_epoch(self):
        return self._num_ex_per_epoch
    
    @property
    def modes(self):
        return ['train', 'val', 'test']

    @property
    def primary_mode(self):
        return 'train'

    def build_feed_dict(self, mode):
        return {}
