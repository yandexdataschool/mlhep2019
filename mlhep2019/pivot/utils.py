import numpy as np
from scipy.stats import entropy

__all__ = [
  'mutual_information',
  'binarize',
  'split'
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


def binarize(xs, n_bins=10):
  if xs.dtype.kind == 'i':
    n = np.max(xs) - np.min(xs) + 1
    indx = xs - np.min(xs)
    return indx, np.arange(n + 1) - 0.5
  else:
    qs = np.linspace(0, 1, num=n_bins + 1)[1:-1]
    bins = np.quantile(xs, qs)
    indx = np.searchsorted(bins, xs)

    return indx, np.concatenate(([np.min(xs)], bins, [np.max(xs)]), axis=0)

def split(*data, split_ratios=0.8, seed=None):
  if len(data) == 0:
    return tuple()

  try:
    iter(split_ratios)
  except TypeError:
    split_ratios = (split_ratios, 1 - split_ratios)

  assert all([r >= 0 for r in split_ratios])

  size = len(data[0])

  split_ratios = np.array(split_ratios)
  split_sizes = np.ceil((split_ratios * size) / np.sum(split_ratios)).astype('int64')
  split_bins = np.cumsum([0] + list(split_sizes))
  split_bins[-1] = size

  if seed is not None:
    state = np.random.get_state()
    np.random.seed(seed)
  else:
    state = None

  r = np.random.permutation(size)

  if state is not None:
    np.random.set_state(state)

  result = list()

  for i, _ in enumerate(split_ratios):
    from_indx = split_bins[i]
    to_indx = split_bins[i + 1]
    indx = r[from_indx:to_indx]

    for d in data:
      if isinstance(d, np.ndarray):
        result.append(d[indx])
      else:
        result.append([ d[i] for i in indx ])

  return tuple(result)
