import argparse
import os

import textwrap
import networkx as nx
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from Overlap_G import Overlap_G
from itertools import combinations
from scipy.special import gamma
from sklearn.decomposition import PCA

import math
import Util
import numpy as np


def analyse(y_predict, y_test):
    accuracy = Util.compute_accuracy(y_predict, y_test)
    return accuracy
    # former method to compute the error
    # error_indices = np.nonzero(y_predict - y_test)[0]
    # error_element = y_test[error_indices]


def rnc(radius, x_training, x_test, y_training, y_test):
    RNC = RadiusNeighborsClassifier(radius=radius)
    RNC.fit(x_training, y_training)
    y_predict = RNC.predict(x_test)
    return analyse(y_predict, y_test)


def rnc_laplace_input(radius, epsilon, x_training, x_test, y_training, y_test):
    max_norm, min_norm = find_max_min_norm(x_training)
    S_f = max_norm - min_norm
    x_training_afterNoise = x_training + np.random.laplace(loc=0, scale=S_f / epsilon, size=x_training.shape)

    RNC = RadiusNeighborsClassifier(radius=radius)
    RNC.fit(x_training_afterNoise, y_training)
    y_predict = RNC.predict(x_test)
    return analyse(y_predict, y_test)


def rnc_laplace(radius, epsilon, x_training, x_test, y_training, y_test):
    RNC = RadiusNeighborsClassifier(radius=radius)
    RNC.fit(x_training, y_training)
    y_predict = Util.predict_laplace_DP(RNC, x_test, epsilon=epsilon / len(x_test))

    return analyse(y_predict, y_test)


def rnc_exponential(radius, epsilon, x_training, x_test, y_training, y_test):
    RNC = RadiusNeighborsClassifier(radius=radius)
    RNC.fit(x_training, y_training)
    y_predict = RNC.predict_exponential_DP(x_test, epsilon=epsilon / len(x_test))

    return analyse(y_predict, y_test)


def rnc_MCS(radius, epsilon, x_training, x_test, y_training, y_test, X_R):
    y_predict = np.zeros(np.shape(y_test))

    if X_R is not None:
        x_test = X_R[:, 0: -1]
        radius = X_R[0, -1]

    # fit the classifier
    RNC = RadiusNeighborsClassifier(radius=radius)
    RNC.fit(x_training, y_training)

    # build region overlap region graph G
    g = Util.build_overlap_region_graph(x_test)
    disconnected_G = list(nx.connected_components(g))

    for g_sub in disconnected_G:
        MCS = Util.compute_MCS(g_sub, g)
        epsilon_prime = epsilon / MCS

        # use vector replacing for loop
        index_in_g_sub = np.array(list(g_sub))
        x_in_g_sub = x_test[index_in_g_sub]
        y_predict[index_in_g_sub] = RNC.predict_laplace_DP(x_in_g_sub, epsilon=epsilon_prime)
    return analyse(y_predict, y_test)


def knn(n_neighbors, epsilon, x_training, x_test, y_training, y_test):
    KNC = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNC.fit(x_training, y_training)
    y_predict = KNC.predict(x_test)
    return analyse(y_predict, y_test)


def knn_exponential(n_neighbors, epsilon, x_training, x_test, y_training, y_test):
    KNC = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNC.fit(x_training, y_training)
    y_predict = KNC.predict_exponential(x_test, epsilon_p=epsilon / len(y_test))
    return analyse(y_predict, y_test)


def find_X_R(k, epsilon_first_part, x_training, x_test, y_training, y_test, candidate_count, RNC):
    # add one col for k and radius
    X_K = np.zeros((x_test.shape[0], x_test.shape[1] + 1))
    X_R = np.zeros((x_test.shape[0], x_test.shape[1] + 1))
    candidates_radius = np.zeros(candidate_count)

    # assign value
    k_neighbor = k
    X_K[:, 0: -1] = x_test
    X_R[:, 0: -1] = x_test
    X_K[:, -1] = k_neighbor

    # construct X_R
    F_dimension = np.shape(x_test)[1] - 1
    vol = Util.compute_vol_omga_D(x_training[:, 0: -1])
    for i in range(len(X_K)):
        k_i = X_K[i, -1]
        r_unit = Util.compute_r_unit(F=F_dimension, D=x_training.shape[0], k=k_i, vol=vol)
        for j in range(len(candidates_radius)):
            candidates_radius[j] = 2 * (j + 1) * r_unit / candidate_count
            RNC.fit(X=x_training, y=y_training)
            neigh_dist, neigh_ind = RNC.radius_neighbors(X=X_K[i, 0: -1].reshape(1, -1), radius=candidates_radius[j])

            sum_neigh_ind = Util.extract_neight_ind(neigh_ind, len(X_K))
            # penalty
            quality_func = -np.fabs(sum_neigh_ind - k_i)

            # find the radius with the bigger probability
            index_max_radius = 0
            max_prob = 0
            prob_j = math.exp(epsilon_first_part * quality_func / (2 * len(X_K)))
            if max_prob < prob_j:
                index_max_radius = j
                max_prob = prob_j
        X_R[i, -1] = candidates_radius[index_max_radius]
    return X_R

def knn_2_rnc_interactive_algorithm(k, radius, epsilon, x_training, x_test, y_training, y_test, candidate_count):
    RNC = RadiusNeighborsClassifier()

    # assign epsilon, weight = 5 for an equal budget share
    weight = 0.5
    epsilon_first_part = weight * epsilon
    epsilon_second_part = (1 - weight) * epsilon

    X_R = find_X_R(k, epsilon_first_part, x_training, x_test, y_training, y_test, candidate_count, RNC)
    Util.test_print("X_R", X_R)
    return rnc_MCS(radius, epsilon_second_part, x_training, x_test, y_training, y_test, X_R)

# todo, it doesn't tell us how to compute m
def knn_non_interactive_algorithm(epsilon, m, gamma):
    x_training, y_training, x_test, y_test = Util.init_iris()

    # add one col for k and radius
    X_K = np.zeros((x_test.shape[0], x_test.shape[1] + 1))
    X_R = np.zeros((x_test.shape[0], x_test.shape[1] + 1))


def PCA_data(epsilon, x_training, y_training, x_test, y_test):
    # sensitivity
    max_norm, min_norm = find_max_min_norm(x_training)
    S_f = max_norm - min_norm
    x_training_afterNoise = x_training + np.random.laplace(loc=0, scale=S_f / epsilon, size=x_training.shape)

    pca = PCA(n_components=2)
    pca.fit(x_training_afterNoise)
    x_test_PCA = pca.transform(x_test)
    x_training_PCA = pca.transform(x_training_afterNoise)
    return x_training_PCA, x_test_PCA


def rnc_PCA(epsilon, x_training, y_training, x_test, y_test):
    x_training, x_test = PCA_data(epsilon, x_training, y_training, x_test, y_test)


def find_max_min_norm(x):
    max_norm, min_norm = 0, 0
    for i in range(len(x)):
        tem = np.linalg.norm(x[i, :])
        if max_norm < tem:
            max_norm = tem
        if min_norm > tem:
            min_norm = tem
    return max_norm, min_norm


def compute_m_grid():
    pass


def run_model(model, epsilon, radius, k, candidate_count):
    # run 10 times to get an average accuracy
    accuracy_sum = 0
    radius, k, candidate_count= Util.check_if_retain_default(radius, k, candidate_count)
    X, y = Util.init_and_preprocess_volcanoe()
    for i in range(10):
        x_training, x_test, y_training, y_test = Util.cross_validation(X, y)
        if model == 1:
            accuracy = rnc(radius, x_training, x_test, y_training, y_test)
        if model == 2:
            accuracy = rnc_laplace_input(radius, epsilon, x_training, x_test, y_training, y_test)
        if model == 3:
            accuracy = rnc_laplace(radius, epsilon, x_training, x_test, y_training, y_test)
        if model == 4:
            accuracy = rnc_exponential(radius, epsilon, x_training, x_test, y_training, y_test)
        if model == 5:
            accuracy = rnc_MCS(radius, epsilon, x_training, x_test, y_training, y_test, None)
        if model == 6:
            accuracy = knn(k, epsilon, x_training, x_test, y_training, y_test)
        if model == 7:
            accuracy = knn_exponential(k, epsilon, x_training, x_test, y_training, y_test)
        if model == 8:
            accuracy = knn_2_rnc_interactive_algorithm(k, radius, epsilon, x_training, x_test, y_training, y_test, candidate_count)
        if model == 9:
            accuracy = rnc_PCA(epsilon, x_training, x_test, y_training, y_test)

        accuracy_sum = accuracy_sum + accuracy
    accuracy = accuracy_sum / 10
    Util.test_print("accuracy", accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classifier with Differential Privacy.'
                                                 'You can select the classifier with DP by the parameter of model.'
                                                 'Each model use cross validation by default.'
                                                 'We use two dataset: iris and volcanoes')
    parser.add_argument('--model', type=int, default=8,
                        help=textwrap.dedent('''
         which model you want to select
         1: Radius Neighbors Classifier(RNC)
         2: Radius Neighbors Classifier(RNC) with Laplace Mechanism on input data
         3: Radius Neighbors Classifier(RNC) with Laplace Mechanism on count
         4: Radius Neighbors Classifier(RNC) with Exponential Mechanism
         5: Radius Neighbors Classifier(RNC) with Exponential Mechanism Sensitivity using Maximum Clique Size
         6: K-Nearst Neighbors Classifier(KNN)
         7: K-Nearst Neighbors Classifier(KNN) with Exponential Mechanism
         8: K-Nearst Neighbors Classifier(KNN) converted to Radius Neighbors Classifier(RNC) with interactive algorithm
         9: K-Nearst Neighbors Classifier(KNN) with Principal Component Analysis(PCA) with Laplace Mechanism'''))
    parser.add_argument('--epsilon', type=float, help='privacy budget', default=2)
    parser.add_argument('--radius', type=float, help='The radius of Radius Neighbors Classifier(RNC).'
                                                     'I already set up the appropriate value in default.'
                                                     'You can just ignore this parameter.'
                                                     'or throw the exception that you can not find points with '
                                                     'this radius'
                        , default=-1)
    parser.add_argument('--k', type=int, help='The k in K-Nearst Neighbors Classifier(KNN)'
                                                'I already set up the appropriate value in default'
                                                'You can just ignore this parameter'
                                                'If you set up a value, it may greatly affect the result'
                        , default=-1)
    parser.add_argument('--candidate_count', type=int, help='the number of candidate in model of'
                                                            'K-Nearst Neighbors Classifier(KNN) converted to '
                                                            'Radius Neighbors Classifier(RNC) with interactive '
                                                            'algorithm.'
                        , default=-1)

    args = parser.parse_args()
    model = args.model
    epsilon = args.epsilon
    radius = args.radius
    k = args.k
    candidate_count = args.candidate_count
    run_model(model, epsilon, radius, k, candidate_count)

    # parser.add_argument('-dataset', type=float, required=True, help='privacy budget')

    # knn_interactive_algorithm(epsilon=epsilon)
    # knn_non_interactive_algorithm(epsilon=epsilon, m=)

    # rnc_laplace(epsilon=epsilon)

    # rnc_MCS(epsilon, None)
