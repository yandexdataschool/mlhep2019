import os

from urllib.request import urlopen
from urllib.parse import urlparse, urljoin

import numpy as np

__all__ = [
  'get_susy'
]

DATA_ROOT_VARS = [
  'DATA_ROOT'
]

SUSY_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'
HIGGS_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'


def get_data_root(root=None):
  import os

  if root is not None:
    return root

  for data_root_var in DATA_ROOT_VARS:
    if data_root_var in os.environ:
      return os.environ[data_root_var]

  return os.path.abspath('./')

def ensure_directory(path):
  if not os.path.exists(path):
    os.makedirs(path)
  else:
    if not os.path.isdir(path):
      raise Exception('%s is present but is not a directory!' % path)
    else:
      pass

def download_and_save(url, path, warn=True):
  if os.path.exists(path):
    raise IOError('Path %s already exists!' % path)

  if warn:
    import warnings
    warnings.warn('Downloading %s to %s'% (url, path))

  response = urlopen(url)
  data = response.read()

  with open(path, 'wb') as f:
    f.write(data)

  return path

def ensure_downloaded(root, *urls, root_url=None, warn=True):
  ensure_directory(root)
  results = []

  for _url in urls:
    if root_url is not None:
      url = urljoin(root_url, _url)
    else:
      url = _url

    path = os.path.join(root, os.path.basename(urlparse(url).path))
    results.append(path)

    if not os.path.exists(path):
      download_and_save(url, path, warn=warn)

  if len(results) == 1:
    return results[0]
  else:
    return tuple(results)

def read_csv_gz(path):
  import gzip

  data = []

  with gzip.open(path, 'r') as f:
    for line in f:
      data.append(
        np.array([ float(x) for x in line.split(b',') ], dtype='float32')
      )

  return np.vstack(data)


def get_csv_gz(url, root=None):
  root = get_data_root(root)
  path = ensure_downloaded(root, url)
  data = read_csv_gz(path)
  return data[:, 1:], data[:, 0]

def get_susy(root=None):
  try:
    f = np.load('SUSY/susy.npz')
    data, labels = f['data'], f['labels']
  except FileNotFoundError:
    data, labels = get_csv_gz(SUSY_URL, root=root)
    np.savez('SUSY/susy.npz', data=data, labels=labels)

  return data, labels
