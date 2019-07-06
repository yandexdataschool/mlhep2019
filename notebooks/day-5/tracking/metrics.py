__author__ = 'mikhail91'

import numpy
import pandas


def predictor(model, X, y):

    event_ids = numpy.unique(y[:, 0])
    y_reco = -1 * numpy.ones(len(y))

    for one_event_id in event_ids:

        mask = y[:, 0] == one_event_id
        X_event = X[mask]
        y_reco_event = model.predict_single_event(X_event)
        y_reco[mask] = y_reco_event

    return y_reco