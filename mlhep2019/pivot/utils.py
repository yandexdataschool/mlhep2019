import numpy as np
from scipy.stats import entropy

__all__ = [
  'mutual_information',
  'binarize'
]

def mutual_information(hist):
  """
  Mutual Information is used as a measure of dependency between variables.

  However, it is inherently difficult to reliably compute MI, thus, don't trust
  this function too much. Also, it is generally not a good idea to use it for
  anything beyond 2 dimensions.

  :param hist: 2D histogram for X and Y random variables.
  :return: rough estimation of mutual information coefficient between X and Y.
  """
  pmf = hist / np.sum(hist, dtype='float64')
  marginal_x, marginal_y = np.sum(pmf, axis=1), np.sum(pmf, axis=0)

  H_xy = entropy(pmf.reshape(-1))

  ### to avoid warnings...
  pmf = np.where(hist > 0, pmf, 1)

  H_x = -np.sum(
    np.where(hist > 0, pmf * (np.log(pmf) - np.log(marginal_x)[:, None]), 0)
  )

  H_y = -np.sum(
    np.where(hist > 0, pmf * (np.log(pmf) - np.log(marginal_y)[None, :]), 0)
  )

  MI = H_xy - H_x - H_y

  return MI

def binarize(xs, n_bins=20):
  if xs.dtype.kind == 'i':
    n = np.max(xs) - np.min(xs) + 1
    indx = xs - np.min(xs)
    return indx, np.arange(n + 1) - 0.5
  else:
    x_min, x_max = np.min(xs), np.max(xs)
    delta = (x_max - x_min) / n_bins

    bins = np.linspace(x_min - 1e-3 * delta, x_max + 1e-3 * delta, num=n_bins)

    return np.searchsorted(bins[1:], xs), bins