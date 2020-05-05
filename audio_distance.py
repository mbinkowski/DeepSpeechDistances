"""
Main Audio distance computation module.
"""

import os
import time
import numpy as np
from glob import glob
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from tensorflow_gan.python.eval.classifier_metrics import kernel_classifier_distance_and_std_from_activations as kernel_dist
from tensorflow_gan.python.eval.classifier_metrics import frechet_classifier_distance_from_activations as frechet_dist
from preprocessing import create_feed_dict


LOAD_PATH = './checkpoint/-54800'
META_PATH = './checkpoint/collection-stripped-meta.meta'

class AudioDistance(object):
  """Main DeepSpeech Distance evaluation class."""

  def __init__(self,
               load_path=LOAD_PATH,
               meta_path=META_PATH,
               keep_features=True,
               required_sample_size=10000,
               num_splits=5,
               do_kdsd=True,
               do_conditional_dsds=True,
               sample_freq=24000):
    """
    args:
      load_path: Path to DeepSpeech2 model checkpoint.
      meta_path: Path to DeepSpeech2 meta graph file.
      keep_features: If True, reference and benchmark features will be kept in
          memory for faster evaluation of future samples.
      required_sample_size: Mimimum sample size required for computation.
          Double of this number of samples is required from reference (real
          data) sample to compute benchmark.
      num_splits: Computation of FDSD and cFDSD will compute mean and std of
          distance based on results from this number of independent runs.
      do_kdsd: If True, Kernel distances (KDSD, cKDSD) will also be computed.
      do_conditional_dsds: If True, conditional distances will be computed.
      sample_freq: Audio sample frequency.
    """
    self.load_path = load_path
    self.meta_path = meta_path

    self.batch_size = 16  # Fixed in DeepSpeech2 graph.
    self.keep_features = keep_features
    self.kept_features = {}
    self.do_kdsd = do_kdsd
    self.do_conditional_dsds = do_conditional_dsds
    self.sample_freq = sample_freq

    self.input_tensors = [
        'IteratorGetNext:0', 'IteratorGetNext:1', 'IteratorGetNext:2']
    self.output_tensor = 'ForwardPass/ds2_encoder/Reshape_2:0'
    self._restored = False

    mult = num_splits * self.batch_size
    if required_sample_size // mult < 1:
        raise ValueError(f"Too small sample size ({required_sample_size}) for "
                         f"given batch size ({self.batch_size}) and number of "
                         f"splits ({num_splits}.")
    self.required_sample_size = (required_sample_size // mult) * mult

    self.saver = tf.train.import_meta_graph(meta_path)

    self.sess_config = tf.ConfigProto(allow_soft_placement=True)
    self.sess_config.gpu_options.allow_growth = True

    shape = (self.required_sample_size, 1600)
    self.ref_features = tf.placeholder(
        tf.float32, shape=shape, name='ref_features')
    self.eval_features = tf.placeholder(
        tf.float32, shape=shape, name='eval_features')

    zipped = zip(tf.split(self.ref_features, num_splits),
                 tf.split(self.eval_features, num_splits))

    dists = [frechet_dist(ref, ev) for ref, ev in zipped]
    self.dists = [(tf.reduce_mean(dists), tf.math.reduce_std(dists))]
    if self.do_kdsd:
      self.dists += [kernel_dist(self.ref_features, self.eval_features,
                                 dtype=tf.float32)]

    self.real_data = None
    self.real_data_benchmarks = None

  def _load_from_pattern(self, pattern, assert_limit=None):
    if assert_limit:
      assert_limit = max(self.required_sample_size, assert_limit)

    def _check_and_cut2limit(x):
      if not assert_limit:
        return x
      if len(x) < assert_limit:
        raise ValueError(
          f"Not enough samples provided ({len(x)}), required: {assert_limit}.")
      return x[:assert_limit]

    if isinstance(pattern, np.ndarray):
      # pattern is already an array
      return _check_and_cut2limit(pattern)

    if isinstance(pattern, list):
      # pattern is a list
      exts = list(np.unique([f[-4:] for f in pattern]))
      if not (len(exts) == 1 and exts[0] in ['.npy', '.wav']):
        raise ValueError("All provided files should be of the same type, "
                         f"either '.npy' or '.wav', got {str(exts)}.")
      files = pattern
    elif isinstance(pattern, str):
      # pattern is a string
      if pattern[-4:] not in ['.npy', '.wav']:
        raise ValueError(f"Wrong filename pattern: {pattern}. Only '.npy' and "
                         "'.wav' files are supported.")
      files = glob(pattern)
    else:
      raise ValueError("Wrong type. Only string, list and arry inputs are "
                       f"supported, got {str(type(pattern))}.")

    if files[0][-4:] == '.npy':
      # npy case
      files_ = []
      for f in files:
        with open(f, 'r') as numpy_file:
          files_.append(np.load(numpy_file))
      array = np.concatenate(files_)
      return _check_and_cut2limit(array)

    # .wav case. Returning a list.
    return _check_and_cut2limit(files)

  def load_real_data(self, pattern):
    """Loads real data from a regex pattern.

    Args:
     pattern: regular expression to locate the data files. Audio needs to be
         stored in .wav or .npy files.
    """
    self.real_data = self._load_from_pattern(
        pattern, assert_limit=2*self.required_sample_size)

  def _restore_graph(self, sess):
    if not self._restored:
      self.saver.restore(sess, self.load_path)
      self._restored = True
      print('Checkpoint restored.')

  def _split_to_batches(self, x):
    bs = self.batch_size
    return [x[k * bs: (k+1) * bs] for k in range(len(x) // bs)]

  def _has_reference_features(self):
    return 'ref' in self.kept_features

  def _has_benchmark_features(self):
    return 'benchmark' in self.kept_features

  def get_features(self, sess=None, files=None):
    """Computes DeepSpeech features for audio from source files.

    Args:
      sess: tf.Session object or None.
      files: None or regex pattern to load the data from. If None, features for
         reference data will be computed.

    Returns:
      numpy array of features for the given data files.
    """
    doing_reference = (files is None)
    if doing_reference:
      # Reference features.
      if self._has_reference_features():
        return self.kept_features['ref']
      # The first half (self.required_sample_size clips) has the same
      # conditioning as the evaluated samples, the second half -- different.
      files = self.real_data
      desc = 'Extracting DeepSpeech features from reference samples'
    else:
      # Evaluated features (which could still be real data).
      files = self._load_from_pattern(files,
                                      assert_limit=self.required_sample_size)
      desc = 'Extracting DeepSpeech features from samples to evaluate'

    features = []

    if sess is None:
      sess = tf.Session(config=self.sess_config)

    self._restore_graph(sess)

    t0 = time.time()
    for idx, file_batch in enumerate(tqdm(self._split_to_batches(files),
                                          desc=desc,
                                          unit_scale=self.batch_size)):
      feed_dict = create_feed_dict(file_batch,
                                   handles=self.input_tensors,
                                   sample_freq=self.sample_freq)
      values = sess.run(self.output_tensor, feed_dict=feed_dict)
      features.append(values.mean(axis=1))

    features_ = np.concatenate(features, axis=0)
    if doing_reference and self.keep_features:
      if not self._has_reference_features():
        # keep reference features for future evaluations
        self.kept_features['ref'] = features_
      if not self._has_benchmark_features():
        # keep benchmark features for future evaluations
        self.kept_features['benchmark'] = np.split(features_, 2)[0]

    print('DeepSpeech2: finished evaluating features, total time'
          '%.1fs', time.time() - t0)
    return features_

  def get_distance(self, sess=None, files=None):
    """Main function computing DeepSpeech distances.
    Args:
      sess: None or tf.Session object.
      files: None or regex pattern with data files to compute distance againts.
          If None, distances for benchmark data will be computed.

    Returns:
      A list of tuples (distance, std) of distances in the following order:
      FDSD, KDSD, cFDSD, cKDSD. If self.do_kdsd is False, Kernel distances will
      be skipped. If do_conditional_dsds is False, conditional distances will
      be skipped.
    """
    doing_real_data_benchmark = (files is None)
    if doing_real_data_benchmark:
      # use the latter part of real wav files for real-data benchmark
      if self.real_data_benchmarks is not None:
        return self.real_data_benchmarks
      elif self._has_benchmark_features() and self._has_reference_features():
        ref_features_ = [self.kept_features['ref']]
        eval_features_ = [self.kept_features['benchmark']]
      else:
        # Evaluate reference features with same conditioning as samples.
        files = self.real_data[:self.required_sample_size]
    else:
      files = self._load_from_pattern(files)

    if sess is None:
      sess = tf.Session(config=self.sess_config)

    if files is not None:
      # Reference features contains 2*self.required_sample_size clips with same
      # and different conditioning.
      ref_features_ = self.get_features(sess=sess, files=None)
      eval_features_ = self.get_features(sess=sess, files=files)

    ref_features_same_cond, ref_features_other_cond = np.split(
        ref_features_, 2)

    print('AudioDistance: got features from both samples, computing '
          'metrics...')
    t0 = time.time()
    dist_vals = sess.run(self.dists,
                         feed_dict={self.ref_features: ref_features_other_cond,
                                    self.eval_features: eval_features_})
    print('AudioDistance: computed metrics from features '
          'in %.1fs.', time.time() - t0)
    if doing_real_data_benchmark:
      self.real_data_benchmarks = dist_vals
      if self.keep_features and (not self._has_benchmark_features()):
        self.kept_features['benchmark'] = eval_features_

    if self.do_conditional_dsds:
      print('Evaluation with the same conditioning.')
      t0 = time.time()
      dist_vals += sess.run(
          self.dists, feed_dict={self.ref_features: ref_features_same_cond,
                                 self.eval_features: eval_features_})
      print('AudioDistance: computed metrics from features '
            'in %.1fs.', time.time() - t0)
    print('AudioDistance: finished evaluation.')
    return dist_vals
