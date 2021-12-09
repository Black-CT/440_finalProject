import os
import warnings
import networkx as nx
import numpy as np
import math
import pandas as pd
from scipy.special import gamma
from sklearn import datasets
from collections import namedtuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# return the disconnected graph as list[set]
import Util

RADIUS_SETUP = 1500.0
N_NEIGHBORS_SETUP = 10
CANDIDATE_COUNT = 1000

def check_if_retain_default(radius, k, candidate_count):
    if radius == -1:
        radius = RADIUS_SETUP
    if k == -1:
        k = N_NEIGHBORS_SETUP
    if candidate_count == -1:
        candidate_count = CANDIDATE_COUNT
    return radius, k, candidate_count


def build_overlap_region_graph(X):
    g = nx.Graph()
    for i in range(len(X)):
        endpoint_inE = judge_overlap(i, X)
        E = build_edge(i, endpoint_inE)
        # add the edge in graph
        for j in range(len(E)):
            g.add_edge(int(E[j, 0]), int(E[j, 1]))
    # create the disconnected graph
    return g


def build_edge(start, end: np):
    E = np.array([])
    for i in end:
        E = np.concatenate((E, np.array([int(start), int(i)])))
    E = E.reshape((-1, 2))
    return E


def judge_overlap(index_i, X):
    index_rest = np.arange(len(X))
    delta_res = delta(X[index_i, 0: -1], X[index_rest, 0: -1])
    radius_sum = X[index_i, -1] + X[index_rest, -1]
    return index_rest[np.where(delta_res <= radius_sum)]


# compute the metric of delta of (x1, x2)
def delta(x1, x2, method="Euclidian"):
    if method == "Euclidian":
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1)).flatten()


# compute the MCS
def compute_MCS(g_sub, g):
    sub_g = g.subgraph(g_sub)
    all_clique = list(nx.find_cliques(sub_g))
    all_clique_size = [len(i) for i in all_clique]
    return max(all_clique_size)


# from stats.py
def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    if a.ndim == 0:
        a = np.atleast_1d(a)
    return a, outaxis


def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly "
                          "checked for nan values. nan values "
                          "will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy


ModeResult = namedtuple('ModeResult', ('mode', 'count'))


def extract_neight_ind(neigh_ind, X):
    for i in neigh_ind:
        res = i

    # outlier_mask = np.zeros(X, dtype=bool)
    # outlier_mask[:] = [len(nind) == 0 for nind in neigh_ind]
    # outliers = np.flatnonzero(outlier_mask)
    # inliers = np.flatnonzero(~outlier_mask)
    # Util.test_print("points in radius:", inliers)
    # if (np.where(outlier_mask != 0)):
    #     raise ValueError(
    #         "No neighbors found for test samples %r, "
    #         "you can try using larger radius, "
    #         "giving a label for outliers, "
    #         "or considering removing them from your dataset." % outliers
    #     )
    return res.size


# the radius of the hypersphere
def compute_r_unit(F, D, k, vol):
    molecular = gamma(F / 2 + 1) * k * vol
    denominator = D * math.pow(math.pi, F / 2)
    return math.pow(molecular / denominator, 1 / F)


# the smallest hyper-cube volume in D
def compute_vol_omga_D(X):
    print("X shape:\n{}".format(X.shape))
    vol = 1
    for feature in range(X.shape[1]):
        vol = vol * (X[:, feature].max() - X[:, feature].min())
    return vol


def test_print(prompt, var):
    print("{}:\n {}".format(prompt, var))


def init_iris():
    iris = datasets.load_iris()
    x_iris = iris.data
    y_iris = iris.target
    return partrition_data(x_iris, y_iris)


def partrition_data(X, y):
    x_training, x_test, y_training, y_test = train_test_split(X, y, test_size=0.1)
    return x_training, x_test, y_training, y_test
    # # the former method to do cross validation
    # # simple cross validation
    # indices = np.random.permutation(len(x_iris))
    #
    # x_training = x_iris[indices[: -10]]
    # y_training = y_iris[indices[: -10]]
    # x_test = x_iris[indices[-10:]]
    # y_test = y_iris[indices[-10:]]


def cross_validation(X, y):
    x_training, x_test, y_training, y_test = partrition_data(X, y)
    return np.array(x_training), np.array(x_test), np.array(y_training), np.array(y_test)


def init_and_preprocess_volcanoe():
    data = pd.read_csv("./data/volcanoes/volcanoes.data", header=None)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(X), y


def compute_accuracy(y_predict, y_test):
    return accuracy_score(y_test, y_predict)


def predict_laplace_DP(rnc, X, epsilon):
    """Predict the class labels for the provided data.

        Parameters
        ----------neigh_ind
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.
        """

    # anyin: return all possible class
    probs = rnc.predict_proba_laplace_DP(X, epsilon)
    classes_ = rnc.classes_

    if not rnc.outputs_2d_:
        probs = [probs]
        classes_ = [rnc.classes_]

    n_outputs = len(classes_)
    n_queries = probs[0].shape[0]
    y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)

    for k, prob in enumerate(probs):
        # iterate over multi-output, assign labels based on probabilities
        # of each output.
        max_prob_index = prob.argmax(axis=1)
        y_pred[:, k] = classes_[k].take(max_prob_index)

        outlier_zero_probs = (prob == 0).all(axis=1)
        if outlier_zero_probs.any():
            zero_prob_index = np.flatnonzero(outlier_zero_probs)
            y_pred[zero_prob_index, k] = rnc.outlier_label_[k]

    if not rnc.outputs_2d_:
        y_pred = y_pred.ravel()

    return y_pred


def predict_proba_laplace_DP(rnc, X, epsilon):
    """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of \
                n_outputs of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

    # anyin: compute the length of x
    n_queries = len(X)

    # anyin: find the neighbors in radius and compute each distances of each points
    # anyin: return distance and corresponding indices in datasets
    neigh_dist, neigh_ind = rnc.radius_neighbors(X)

    # anyin: because some samples may don't have neighbors within radius, we separate them here
    outlier_mask = np.zeros(n_queries, dtype=bool)
    outlier_mask[:] = [len(nind) == 0 for nind in neigh_ind]
    outliers = np.flatnonzero(outlier_mask)
    inliers = np.flatnonzero(~outlier_mask)

    classes_ = rnc.classes_
    _y = rnc._y
    if not rnc.outputs_2d_:
        _y = rnc._y.reshape((-1, 1))
        classes_ = [rnc.classes_]

    if rnc.outlier_label_ is None and outliers.size > 0:
        raise ValueError(
            "No neighbors found for test samples %r, "
            "you can try using larger radius, "
            "giving a label for outliers, "
            "or considering removing them from your dataset." % outliers
        )
    probabilities = []
    # iterate over multi-output, measure probabilities of the k-th output.
    for k, classes_k in enumerate(classes_):
        pred_labels = np.zeros(len(neigh_ind), dtype=object)
        pred_labels[:] = [_y[ind, k] for ind in neigh_ind]

        proba_k = np.zeros((n_queries, classes_k.size))
        proba_inl = np.zeros((len(inliers), classes_k.size))

        # samples have different size of neighbors within the same radius
        for i, idx in enumerate(pred_labels[inliers]):
            proba_inl[i, :] = np.bincount(idx, minlength=classes_k.size)
        # anyin: noise1_laplace
        # anyin: add noise here, then replace the former
        # todo add the initial value
        proba_k[inliers, :] = proba_inl + np.random.laplace(loc=0, scale=1 / epsilon, size=proba_inl.shape)
        # anyin: former one
        # proba_k[inliers, :] = proba_inl

        if outliers.size > 0:
            _outlier_label = rnc.outlier_label_[k]
            label_index = np.flatnonzero(classes_k == _outlier_label)
            if label_index.size == 1:
                proba_k[outliers, label_index[0]] = 1.0
            else:
                warnings.warn(
                    "Outlier label {} is not in training "
                    "classes. All class probabilities of "
                    "outliers will be assigned with 0."
                    "".format(rnc.outlier_label_[k])
                )

        # normalize 'votes' into real [0,1] probabilities
        normalizer = proba_k.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba_k /= normalizer

        probabilities.append(proba_k)

    if not rnc.outputs_2d_:
        probabilities = probabilities[0]

    return probabilities

