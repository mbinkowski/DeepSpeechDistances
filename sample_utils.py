"""Utility file to subsample random clips from longer audio file."""

from scipy.io.wavfile import read, write
import os
import numpy as np
from tqdm import tqdm


def _mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)

def subsample_audio(file_path, sample_path, num_samples=1000,
                    num_noise_levels=3, length=2):
  """Helper sampling function.

  Args:
    file_path: Path to the source audio file.
    sample_path: Path to save subsamples in.
    num_samples: Numer of clips to sample from the source file.
    num_noise_levels: Number of noise levels. Apart from clean samples,
       this number of noisy versions of each sampled clip will be saved, with
       noise levels chosen from logspace between 10^-3 and 10^-1.5.
    length: Length of subsampled clips, in seconds.
  """
  freq, base_wav = read(file_path)
  base_wav = base_wav.astype(np.float32) / 2**15
  length *= freq

  start = np.random.randint(0, base_wav.shape[0] - length + 1,
                            size=(2 * num_samples, ))
  noise_levels = np.logspace(-3, -1, num_noise_levels)

  _mkdir(sample_path)
  _mkdir(os.path.join(sample_path, 'ref'))
  for i in range(num_noise_levels):
    _mkdir(os.path.join(sample_path, f'noisy_{i + 1}'))

  for k, start_k in enumerate(tqdm(start, desc='Saving audio sample files')):
    window = base_wav[start_k: start_k + length]
    write(os.path.join(sample_path, 'ref', '%05d.wav' % (k + 1)), freq, window)

    for i, noise_level in enumerate(noise_levels):
      noisy_window = window + np.random.normal(scale=noise_level, size=(length, ))
      noisy_window = np.clip(noisy_window * 2 ** 15, -2 ** 15, 2 ** 15 - 1)
      write(os.path.join(sample_path, f'noisy_{i + 1}', '%05d.wav' % (k + 1)),
            freq, noisy_window.astype(np.int16))
