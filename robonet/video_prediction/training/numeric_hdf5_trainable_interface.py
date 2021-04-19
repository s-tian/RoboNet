from robonet.video_prediction.training.trainable_interface import VPredTrainable


class HDF5TrainableInterface(VPredTrainable):

    def _init_sources(self):
        return self._batch_config[0]['data_directory'], None
