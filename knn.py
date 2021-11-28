from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt


def knn(x_iris, y_iris):

    # can be changed into stratified cross validation
    # todo 1.cross validation
    # todo 2.normalization
    indices = np.random.permutation(len(x_iris))

    x_train_iris = x_iris[ indices[ 0 : -10 ]]
    y_train_iris = y_iris[ indices[ 0 : -10 ]]
    x_test_iris = x_iris[ indices[ -10 :  ]]
    y_test_iris = y_iris[ indices[ -10 : ]]

    knn = KNeighborsClassifier(n_neighbors=3)
    # todo 里面有参数可以调整 度量距离的方式，最好的neighbor的用法

    knn.fit(x_train_iris, y_train_iris)
    y_predict = knn.predict(x_test_iris)
    print("-----")
    print(y_predict)
    print(y_test_iris)

    # take the the value of line 0
    error_index = np.nonzero(y_predict - y_test_iris)[0]
    print(error_index)

def test_laplace():
    # Laplace
    # loc is the location parameter, like bias
    # scale is the scale parameter. It's epsilon in DP
    a = laplace.pdf(0, loc = 0, scale = 1)
    print(a)

    x = np.linspace( -10, 10, 200 )
    y = laplace.pdf(x, loc = 0, scale = 1)
    fig, ax = plt.subplots( 1, 1 )
    ax.plot( x, y )
    plt.show()


if __name__ == "__main__":


    # todo 1。 直接在输出加 epsilon
    pass
    # logic control part:

'''
    iris = datasets.load_iris()
    x_iris = iris.data
    y_iris = iris.target

    print(x_iris)

    dimension = np.shape(x_iris)
    knn(x_iris, y_iris)
'''


# 有一个画knn决策边界的demo，来自官网。 但是我的项目可能用不上，因为维度超过二维了，所以大概率用不上
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py





