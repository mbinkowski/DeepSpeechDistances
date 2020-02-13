"""Data preprocessing functions for DeepSpeech distances.

Based on NVIDIA's OpenSeq2Seq's code:
https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech2text.py
https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_utils.py
"""

import io
import os
import scipy.io.wavfile
import numpy as np
import tensorflow.compat.v1 as tf
import resampy as rs
import python_speech_features as psf


def normalize_signal(signal):
  """
  Normalize float32 signal to [-1, 1] range
  """
  return signal / (np.max(np.abs(signal)) + 1e-5)


def get_speech_features(signal,
                        sample_freq,
                        num_features=160,
                        pad_to=8,
                        window_size=20e-3,
                        window_stride=10e-3,
                        base_freq=16000):
  """Function to convert raw audio signal to numpy array of features.
  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (int): Frames per second.
    num_features (int): Number of speech features in frequency domain.
    pad_to (int): If specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    window_size (float): Size of analysis window in milli-seconds.
    window_stride (float): Stride of analysis window in milli-seconds.
    base_freq (int): Frequency at which spectrogram will be computed.

  Returns:
    Tuple of np.array of audio features with shape=[num_time_steps,
    num_features] and duration of the signal in seconds (float).
  """
  signal = signal.astype(np.float32)

  if sample_freq != base_freq:
    signal = rs.resample(signal, sample_freq, base_freq, filter='kaiser_best')
    sample_freq = base_freq
  
  signal = normalize_signal(signal)

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)

  length = 1 + int(np.ceil(
      (1.0 * signal.shape[0] - n_window_size) / n_window_stride))
  if pad_to > 0:
    if length % pad_to != 0:
      pad_size = (pad_to - length % pad_to) * n_window_stride
      signal = np.pad(signal, (0, pad_size), mode='constant')

  frames = psf.sigproc.framesig(sig=signal,
                                frame_len=n_window_size,
                                frame_step=n_window_stride,
                                winfunc=np.hanning)

  features = psf.sigproc.logpowspec(frames, NFFT=n_window_size)
  if num_features > n_window_size // 2 + 1:
    raise ValueError(
       f"num_features (= {num_features}) for spectrogram should be <= (sample_"
       f"freq (= {sample_freq}) * window_size (= {window_size}) // 2 + 1)")

  # cut high frequency part
  features = features[:, :num_features]

  if pad_to > 0:
    assert features.shape[0] % pad_to == 0
  mean = np.mean(features)
  std_dev = np.std(features)
  features = (features - mean) / std_dev
  features = features.astype(np.float16)

  return features, audio_duration


def get_audio_tuple(inputs, sample_freq=24000., dtype=tf.float16, **kwargs):
  """Parses audio from wav and returns a tuple of (audio, audio length).
  Args:
    inputs: numpy array containing waveform or a wav file name.
    sample_freq: Default audio frequency; ignored if wav fiel is passed.
    dtype: Data type for audio array.
    **kwargs: Additional arguments to be passed to get_speech_features.

  Returns:
    tuple: source audio features as ``np.array``, length of source sequence.
  """
  if isinstance(inputs, str):
    sample_freq, signal = scipy.io.wavfile.read(open(inputs, 'rb'))
  elif isinstance(inputs, np.ndarray):
    signal = inputs
  else:
    raise ValueError(
        f"Only string or numpy array inputs are supported. Got {type(line)}")

  source, audio_duration = get_speech_features(signal, sample_freq, **kwargs)

  return source.astype(dtype.as_numpy_dtype()), np.int32([len(source)])


def create_feed_dict(model_in, handles=None, num_audio_features=160, **kwargs):
  """ Creates the feed dict for DeepSpeech distance computation.

  Args:
    model_in (str or np.array): Either a str that contains the file path of the
        wav file, or a numpy array containing 1-d wav file.
    handles: List of Tensor/placeholder names for data to be fed to. If None,
        a list will be returned.
    num_audio_features: Number of spectrogram features to be extracted.
    **kwargs: Additional keyword arguments to be passed to get_audio_tuple.

  Returns:
    feed_dict (dict): Dictionary with values for the placeholders, or a list
    of values if no 'handles' argument is passed.
  """
  audio_arr, audio_length_arr = [], []

  for line in model_in:
    audio, audio_length  = get_audio_tuple(
        line, num_features=num_audio_features, **kwargs)
    audio_arr.append(audio)
    audio_length_arr.append(audio_length)

  max_len = np.max(audio_length_arr)

  for i, audio in enumerate(audio_arr):
    if max_len > len(audio):
      audio = np.pad(
          audio, ((0, max_len - len(audio)), (0, 0)),
          "constant", constant_values=0.)
      audio_arr[i] = audio

  batch_size = len(model_in)
  audios = np.reshape(audio_arr, [batch_size, -1, num_audio_features])
  audio_lengths = np.reshape(audio_length_arr, [batch_size, 1])
  ids = np.zeros((batch_size, 1), dtype=np.int32)

  if handles is None:
    return (audios, audio_lengths, ids)

  return dict(zip(handles, [audios, audio_lengths, ids]))
