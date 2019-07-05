import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from importlib import import_module

def score_function(y_true, y_pred):
    '''Compute a clustering score.

    Cluster ids should be nonnegative integers. A negative integer
    will mean that the corresponding point does not belong to any
    cluster.

    We first identify assigned clusters by taking the max count of 
    unique assigned ids for each true cluster. We remove all unassigned
    clusters (all assigned ids are -1) and all duplicates (the same
    assigned id has majority in several true clusters) except the one
    with the largest count. We add the counts, then divide by the number
    of events. The score should be between 0 and 1. 

    Parameters
    ----------
    y_true : np.array, shape = (n, 2)
        The ground truth.
        first column: event_id
        second column: cluster_id
    y_pred : np.array, shape = n
        The predicted cluster assignment (predicted cluster_id)
    """
    '''
    score = 0.
    event_ids = y_true[:, 0]
    y_true_cluster_ids = y_true[:, 1]
    y_pred_cluster_ids = y_pred

    unique_event_ids = np.unique(event_ids)
    for event_id in unique_event_ids:
        event_indices = (event_ids==event_id)
        cluster_ids_true = y_true_cluster_ids[event_indices]
        cluster_ids_pred = y_pred_cluster_ids[event_indices]

        unique_cluster_ids = np.unique(cluster_ids_true)
        n_cluster = len(unique_cluster_ids)
        n_sample = len(cluster_ids_true)

        # assigned_clusters[i] will be the predicted cluster id
        # we assign (by majority) to true cluster i 
        assigned_clusters = np.empty(n_cluster, dtype='int64')
        # true_positives[i] will be the number of points in 
        # predicted cluster[assigned_clusters[i]]
        true_positives = np.full(n_cluster, fill_value=0, dtype='int64')
        for i, cluster_id in enumerate(unique_cluster_ids):
            # true points belonging to a cluster
            true_points = cluster_ids_true[cluster_ids_true == cluster_id]
            # predicted points belonging to a cluster
            found_points = cluster_ids_pred[cluster_ids_true == cluster_id]
            # nonnegative cluster_ids (negative ones are unassigned)
            assigned_points = found_points[found_points >= 0]
            # the unique nonnegative predicted cluster ids on true_cluster[i]
            n_sub_cluster = len(np.unique(assigned_points))
            # We find the largest predicted cluster in the true cluster.
            if(n_sub_cluster > 0):
                # sizes of predicted assigned cluster in true cluster[i]
                predicted_cluster_sizes = np.bincount(
                    assigned_points.astype(dtype='int64'))
                # If there are ties, we assign the tre cluster to the predicted
                # cluster with the smallest id (combined behavior of np.unique
                # which sorts the ids and np.argmax which returns the first 
                # occurence of a tie).
                assigned_clusters[i] = np.argmax(predicted_cluster_sizes)
                true_positives[i] = len(
                    found_points[found_points == assigned_clusters[i]])
            # If none of the assigned ids are positive, the cluster is unassigned
            # and true_positive = 0
            else:
                assigned_clusters[i] = -1
                true_positives[i] = 0

        # resolve duplicates and count good assignments
        sorted = np.argsort(true_positives)
        true_positives_sorted = true_positives[sorted]
        assigned_clusters_sorted = assigned_clusters[sorted]
        good_clusters = assigned_clusters_sorted >= 0
        for i in range(len(assigned_clusters_sorted) - 1):
            assigned_cluster = assigned_clusters_sorted[i]
            # duplicates: only keep the last count (which is the largest
            # because of sorting)
            if assigned_cluster in assigned_clusters_sorted[i+1:]:
                good_clusters[i] = False
        n_good = np.sum(true_positives_sorted[good_clusters])
        score += 1. * n_good / n_sample
    score /= len(unique_event_ids)
    return score


filename = 'public_train.csv'


def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['event_id', 'cluster_id']]
    X_df = df.drop(['cluster_id'], axis=1)
    return X_df.values, y_df.values


def train_submission(module_path, X_array, y_array, train_is):
    clusterer = import_module('clusterer', module_path)
    ctr = clusterer.Clusterer()
    ctr.fit(X_array[train_is], y_array[train_is])
    return ctr


def test_submission(trained_model, X_array, test_is):
    ctr = trained_model
    X = X_array[test_is]
    unique_event_ids = np.unique(X[:, 0])
    cluster_ids = np.empty(len(X), dtype='int')

    for event_id in unique_event_ids:
        event_indices = (X[:, 0] == event_id)
        # select an event and drop event ids
        X_event = X[event_indices][:, 1:]
        cluster_ids[event_indices] = ctr.predict_single_event(X_event)

    return np.array(cluster_ids)


# We do a single fold because blending would not work anyway:
# mean of cluster_ids make no sense
def get_cv(y_train_array):
    unique_event_ids = np.unique(y_train_array[:, 0])
    event_cv = ShuffleSplit(
        n_splits=1, test_size=0.5, random_state=57)
    for train_event_is, test_event_is in event_cv.split(unique_event_ids):
        train_is = np.where(
            np.in1d(y_train_array[:, 0], unique_event_ids[train_event_is]))[0]
        test_is = np.where(
            np.in1d(y_train_array[:, 0], unique_event_ids[test_event_is]))[0]
        yield train_is, test_is


if __name__ == '__main__':
    print("Reading file ...")
    X, y = read_data(filename)
    unique_event_ids = np.unique(X[:, 0])
    cv = get_cv(y)
    for train_is, test_is in cv:
        print("Training ...")
        trained_model = train_submission('', X, y, train_is)
        print("Testing ...")
        y_pred = test_submission(trained_model, X, test_is)
        score = score_function(y[test_is], y_pred)
        print ('score = ', score)

