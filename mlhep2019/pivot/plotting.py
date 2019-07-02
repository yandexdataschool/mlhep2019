import numpy as np
import torch

import matplotlib.pyplot as plt

__all__ = [
  'plot_losses',

  'make_grid',
  'draw_response',

  'nuisance_prediction_hist',
  'nuisance_metric_plot'
]

def make_grid(data, size=25):
  data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
  delta = data_max - data_min

  xs = np.linspace(data_min[0] - 0.05 * delta[0], data_max[0] + 0.05 * delta[0], num=size)
  ys = np.linspace(data_min[1] - 0.05 * delta[1], data_max[1] + 0.05 * delta[1], num=size)

  grid_X, grid_Y = np.meshgrid(xs, ys)

  grid = np.stack([grid_X.reshape(-1), grid_Y.reshape(-1)]).astype('float32').T
  return xs, ys, grid

def draw_response(xs, ys, probabilities, data, labels):
  probabilities = probabilities.reshape(ys.shape[0], xs.shape[0])

  plt.contourf(xs, ys, probabilities, levels=np.linspace(0, 1, num=20), cmap=plt.cm.plasma)

  plt.scatter(data[labels > 0.5, 0], data[labels > 0.5, 1])
  plt.scatter(data[labels < 0.5, 0], data[labels < 0.5, 1])

  plt.contour(
    xs, ys, probabilities, levels=[0.25, 0.5, 0.75],
    colors=['black', 'black', 'black'],
    linewidths=[3, 3, 3],
    linestyles=['dashed', 'solid', 'dashed']
  )

def nuisance_prediction_hist(predictions, nuisance, labels, names=None, nuisance_bins=10, prediction_bins=10):
  """
  Plots distribution of predictions against nuisance parameter.

  :param predictions: a tuple of 1D numpy arrays containing predictions of models;
  :param nuisance: a 1D numpy array containing values of the nuisance parameter;
  :param labels: a 1D numpy array with labels;
  :param names: names of the models;
  :param nuisance_bins: number of bins for the nuisance parameter, ignored if `nuisance` is an array of an integer type;
  :param prediction_bins: number of bins for the predictions.
  :return:
  """
  from .utils import mutual_information, binarize
  indx, nu_bins = binarize(nuisance, n_bins=nuisance_bins)

  p_bins = np.linspace(0, 1, num=prediction_bins)

  hists = [
    np.histogram2d(proba, nuisance, bins=(p_bins, nu_bins))[0]
    for proba in predictions
  ]

  n_classes = np.max(labels) + 1

  class_hists = [
    [
      np.histogram2d(proba[labels == y], nuisance[labels == y], bins=(p_bins, nu_bins))[0]
      for y in range(n_classes)
    ] for proba in predictions
  ]

  max_v = max([ np.max(hist) for hist in hists ])

  plt.subplots(nrows=nuisance_bins, ncols=len(predictions), figsize=(5 * len(predictions), 3 * nuisance_bins))

  for j, _ in enumerate(predictions):
    mi = mutual_information(hists[j])

    class_mis = [
      mutual_information(class_hists[j][k])
      for k in range(n_classes)
    ]

    for i in range(nuisance_bins):
      plt.subplot(nuisance_bins, len(predictions), i * len(predictions) + j + 1)
      plt.step(
        (p_bins[1:] + p_bins[:-1]) / 2,
        hists[j][:, nuisance_bins - i - 1],
        label=(None if i != 0 else 'total'),
        where='mid'
      )
      for y in range(n_classes):
        plt.step(
          (p_bins[1:] + p_bins[:-1]) / 2,
          class_hists[j][y][:, nuisance_bins - i - 1],
          label=(None if i != 0 else ('class %d' % (y, ) )),
          where='mid'
        )

      plt.ylim([0, 1.05 * max_v])

      if i == 0:
        mis_str = ', '.join([ '$\mathrm{MI}_%d=\mathrm{%.2lf}$' % (k, x) for k, x in enumerate(class_mis) ])
        if names is None:
          name = 'Model %d' % (j, )
        else:
          name = names[j]

        plt.title(
          '%s, $\mathrm{MI}=\mathrm{%.2lf}$\n%s' % (
            name, mi, mis_str
          )
        )

        plt.legend()

def nuasance_predictions_plot(model, data, nuisance):
  from .utils import mutual_information

  p = torch.sigmoid(model(data)).cpu().detach().numpy()

  p_bins = np.linspace(np.min(p), np.max(p), num=21)
  nu_bins = np.arange(6)
  hist, _, _ = np.histogram2d(p, nuisance, bins=(p_bins, nu_bins))
  pmf = hist / np.sum(hist)

  mi = mutual_information(hist)

  plt.imshow(pmf.T, extent=[np.min(p), np.max(p), 0, 5], aspect='auto')
  plt.title('MI = %.2e' % (mi,))
  plt.colorbar()
  plt.xlabel('Network predictions')
  plt.ylabel('Nuisance parameter')

def nuisance_metric_plot(predictions, labels, nuisance, metric_fn, base_level=0.5, names=None, nuisance_bins=10):
  from .utils import binarize

  indx, nu_bins = binarize(nuisance, nuisance_bins)

  metrics = []

  labels_binned = [
    list()
    for _ in range(nu_bins.shape[0] - 1)
  ]
  for i in range(labels.shape[0]):
    labels_binned[indx[i]].append(labels[i])

  labels_binned = [
    np.where(np.array(ls) > 0.5, 1, 0)
    for ls in labels_binned
  ]

  for proba in predictions:
    proba_binned = [
      list()
      for _ in range(nu_bins.shape[0] - 1)
    ]

    for i in range(proba.shape[0]):
      bin_i = indx[i]
      proba_binned[bin_i].append(proba[i])

    proba_binned = [ np.array(ps, dtype='float32') for ps in proba_binned ]

    metric = np.array([
      metric_fn(ls, np.array(pred))
      for pred, ls  in zip(proba_binned, labels_binned)
    ])

    metrics.append(metric)

  for i, metric in enumerate(metrics):
    plt.plot(
      (nu_bins[:-1] + nu_bins[1:]) / 2, metric,
      label=(names[i] if names is not None else None)
    )

  m_min = min([ np.min(metric) for metric in metrics ])
  m_max = max([ np.max(metric) for metric in metrics ])
  m_delta = m_max - m_min

  plt.ylim([min(m_min, base_level), m_max + 0.05 * m_delta ])
  plt.ylabel('metric')
  plt.xlabel('nuisance parameter')

  if names is not None:
    plt.legend()

def plot_losses(epoch, **kwargs):
  plt.figure(figsize=(9, 4))

  plt.xlabel('epoch')
  plt.ylabel('loss')
  for i, (name, losses) in enumerate(kwargs.items()):
    xs = np.arange(epoch + 1)

    mean = np.mean(losses[:(epoch + 1)], axis=1)
    std = np.std(losses[:(epoch + 1)], axis=1)

    color = plt.cm.Set1(i)
    plt.plot(xs, mean, label=name, color=color)
    plt.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)

  n_epoches = max([ losses.shape[0] for losses in kwargs.values() ])
  plt.xlim([0, n_epoches - 1])

  plt.legend()
