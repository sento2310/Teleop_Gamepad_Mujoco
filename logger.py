import h5py
import numpy as np
import os


class Logger:
    def __init__(self, filename, buffer_size=1000, feature_names=None,
                 feature_dtype='f8', label_dtype='i', ts_dtype='f8'):
        self.filename = filename
        self.buffer_size = max(1, int(buffer_size))
        self.feature_names_init = feature_names
        self.feature_dtype = np.dtype(feature_dtype)
        self.label_dtype = np.dtype(label_dtype)
        self.ts_dtype = np.dtype(ts_dtype)
        self.file = h5py.File(filename, 'a', libver='latest')
        self._feature_names = None
        self.feature_count = None
        self.initialized = False
        self._buffer = {'timestamps': [], 'features': [], 'labels': []}
        self._prepare_datasets()

        self.features = 0
        self.timestamp = 0
        self.label = 0


    @property
    def feature_names(self):
        if self._feature_names is None and 'features' in self.file:
            names = self.file['features'].attrs.get('feature_names')
            if names is not None:
                self._feature_names = [
                    n.decode('utf-8') if isinstance(n, bytes) else str(n)
                    for n in names
                ]
        return self._feature_names

    def _prepare_datasets(self):
        # If all three datasets already exist, mark the logger as initialized.
        if all(name in self.file for name in ['timestamps', 'features', 'labels']):
            ds_feat = self.file['features']
            self.feature_count = ds_feat.shape[1]
            self._feature_names = self.feature_names  # load stored names
            self.initialized = True

    def _init_datasets(self, features_example):
        # Determine the number of features from the first data point.
        self.feature_count = len(features_example)
        if self.feature_names_init and len(self.feature_names_init) == self.feature_count:
            names = self.feature_names_init
        else:
            names = [f'feature_{i}' for i in range(self.feature_count)]
            print("WARNING: feature do not fit")
        self._feature_names = names
        # Define simple chunk shapes.
        ts_chunk = (min(self.buffer_size * 5, 16384),)
        ft_chunk = (
        min(self.buffer_size * 5, max(1, 1024 * 1024 // (self.feature_count * self.feature_dtype.itemsize))),
        self.feature_count)
        lb_chunk = (min(self.buffer_size * 5, 16384),)

        self.file.create_dataset(
            'timestamps', shape=(0,), maxshape=(None,),
            dtype=self.ts_dtype, chunks=ts_chunk, compression='gzip'
        )
        ds_feat = self.file.create_dataset(
            'features', shape=(0, self.feature_count), maxshape=(None, self.feature_count),
            dtype=self.feature_dtype, chunks=ft_chunk, compression='gzip'
        )
        ds_feat.attrs['feature_names'] = [n.encode('utf-8') for n in names]
        self.file.create_dataset(
            'labels', shape=(0,), maxshape=(None,),
            dtype=self.label_dtype, chunks=lb_chunk, compression='gzip'
        )
        self.initialized = True

    def log(self):
        if self.file is None:
            raise ValueError("File is closed.")
        features_array = np.asarray(self.features, dtype=self.feature_dtype)
        if not self.initialized:
            self._init_datasets(features_array)

        if features_array.shape != (self.feature_count,):
            raise ValueError(f"Features must have shape ({self.feature_count},) but got {features_array.shape}.")

        self._buffer['timestamps'].append(self.ts_dtype.type(self.timestamp))
        self._buffer['features'].append(features_array)
        self._buffer['labels'].append(self.label_dtype.type(self.label))

        if len(self._buffer['timestamps']) >= self.buffer_size:
            self.flush()

    def flush(self):
        if self.file is None or not self.initialized or not self._buffer['timestamps']:
            return

        start = self.file['timestamps'].shape[0]
        n = len(self._buffer['timestamps'])
        new_size = start + n

        self.file['timestamps'].resize(new_size, axis=0)
        self.file['features'].resize(new_size, axis=0)
        self.file['labels'].resize(new_size, axis=0)

        self.file['timestamps'][start:new_size] = np.array(self._buffer['timestamps'], dtype=self.ts_dtype)
        self.file['features'][start:new_size, :] = np.stack(self._buffer['features'], axis=0)
        self.file['labels'][start:new_size] = np.array(self._buffer['labels'], dtype=self.label_dtype)

        self._buffer = {'timestamps': [], 'features': [], 'labels': []}
        self.file.flush()

    def close(self):
        if self.file:
            self.flush()
            self.file.close()
            self.file = None
            self.initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ----- Usage Example -----
if __name__ == "__main__":
    log_filename = 'robot_log_simple.h5'
    robot_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    num_features = len(robot_joint_names)

    # Remove existing file for a clean test.
    if os.path.exists(log_filename):
        os.remove(log_filename)

    print("Writing data...")
    with Logger(log_filename, buffer_size=5, feature_names=robot_joint_names) as logger:
        for i in range(7):
            ts = 100.0 + i * 0.1
            features = np.round(np.sin(np.arange(num_features) + (i * 0.5)), 3)
            label = i % 2
            logger.label = label
            logger.features = features
            logger.log()
            print(f"Logged point {i}")

    print("Data written.\nReading data...")
    with h5py.File(log_filename, 'r') as f:
        timestamps = f['timestamps'][:]
        features = f['features'][:]
        labels = f['labels'][:]
        print("Timestamps:", timestamps)
        print("Features:\n", features)
        print("Labels:", labels)
        print("feature names", f['features'].attrs.get('feature_names'))
    print("Finished.")