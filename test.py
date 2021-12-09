import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Util
import math


def test0():
    a = np.array([[1,2,3,4],
                  [1,2,3,4]])
    print(a)

    # 0 is the most frequency type user use
    sum0_row = a.sum(axis=0)
    sum1_row = a.sum(axis=1)

    print(sum0_row)
    print(sum1_row)

    print(np.newaxis)
    print(type(np.newaxis))


    print("--test enumerate---")
    b = ["a", "b", "c"]
    for index, value in enumerate(b):
        print(index, value)
        print("haha")

# one short method write if statement and for loop
def test1():
    neigh_ind1 = [[1,2,3,4]]
    ans = [len(nind) == 0 for nind in neigh_ind1]
    print(ans)

def test2():
    # x中最大的数位5，所以bin的值为6，现在我们有一个范围，0-5的正整数，在这个x中出现的频率
    x = [1,2,3,4,4,3,2,5]
    # because value 0 doesn't exist, so the value is 0
    # 因为1出现了一次，所以记1
    # 2出现了2次，所以记2
    ans = np.bincount(x)
    print(ans)

    # when it comes to minlength parameter
    # this parameter specify the minimum value of bin,
    # if the bin of this array is smaller than minlength, we use minlength.
    # if the bin of this array is bigger than minlength, we use the bin.
    ans2 = np.bincount(x, minlength=7)
    print(ans2)
    ans3 = np.bincount(x, minlength=1)
    print(ans3)

def test3():
    # you can omit the loop in numpy by this way
    # judge sentences can be written in []
    a = np.array([1,2,3,4,5,6,0,0,0])
    a[ a == 0 ] = 1
    print(a)

def test4():
    a = np.array([[1,2,3,4],
                  [1,2,3,4]])
    a[a==4] = np.exp(a)
    print(a)

# plot the laplace sequential composition
def test5():
    # x = np.linspace(-10, 10, 1000)
    epsilton = 1
    # y = stats.laplace.pdf(x, loc = 0, scale = epsilton)
    # plt.plot(x,y)
    # plt.title("epsilon = " + str(epsilton))
    # plt.show()

    x = np.linspace(-10, 10, 1000)
    for i in range(0,1000):
        # get one sample
        noise = np.random.laplace(loc = 0, scale = epsilton)
        plt.scatter(x[i], noise)
    plt.show()

def test6():
    # sample multiplely
    # 1. sample from laplace distribution
    # set up epsilon and mean
    mean = 0
    epsilon = 1

    # set up target which we add noise on
    # we first try 2 dimensional array, 1 dimensional array have problem when we create laplace noise
    x = np.zeros((4, 4))

    # ps: x.shape return the number of row and col
    # ps: x.size return the number of elements

    # np.random.laplace can sample from laplace distribution and generate a numpy data
    noise = np.random.laplace(size = x.shape, loc = mean, scale = epsilon)
    y = x + noise
    print(y)

    # now it comes to 1 dimensional array
    x1 = np.zeros((1,10))
    noise1 = np.random.laplace(size = x1.shape, loc = mean, scale = epsilon)
    y2 = x1 + noise1
    print(y2)

def test7():
    # len function returns the number of row of the target
    a = np.array([[1,2,3], [1,2,3]])
    print(len(a))

# train the method of utilizing object
class Teacher():
    # the variable/attribute of class
    stuff_number = np.array([1,2,3,4,5,6])

    # the variable of instance should be defined in __init__
    def __init__(self, name, age, subject):
        self.name = name
        self.age = age
        self.subject = subject

    def print_stuff(self):
        print(Teacher.stuff_number)
    
def test8():
    t1 = Teacher("123",1,1)
    t1.print_stuff()


from Overlap_G import Overlap_G
def test9():
    G1 = Overlap_G(None, None, None, np.array([1,2,3]))
    a = np.array([1,2])
    G1.add_G(a, np.array([1,2]),  G1, np.array([1,2]) )

from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier
from itertools import combinations

# method list(method())
def test14():
    a = [1,2,3]
    sol = list(combinations(a,2))
    i = sol[1]
    print(type(i))
    print(sol)
    print(type(sol))


# test the add function of numpy
def test13():
    a = []
    a.append(1)
    b = []
    b.append([1,2,3])
    print(a)
    print(b)
    a = np.array(a)
    print(type(a))
    b = np.array(b)
    print(type(b))

# how to judge if one list is null
def test11():
    a = []
    print(len(a))

    # don't enter this cycle
    if len(a):
        print("list is null")

'''
    X: test instances
'''
def test10(X, radius):
    X = X
    neigh = NearestNeighbors(radius=radius)
    neigh.fit(X)
    NearestNeighbors(radius=1.5)
    A = neigh.radius_neighbors_graph(X)
    print(A)
    print(type(A))
    print("---")
    print(A.toarray())
    print(type(A.toarray()))

    incidence_matrix = A.toarray()
    row, col = np.shape(incidence_matrix)
    print("row, col", row, col)
    # initialize G
    G = Overlap_G(V = np.arange(len(X)), E = None, X = X)
    # create overlap region graph
    for i in range(row):
        print("---new row---")
        # if one row is already used, we must find all overlap region graph of it
        # so j start from i to columns
        # find the overlap region graph in one row and only can find in one row
        for j in range(i+1, col):
            V = []
            E = []
            X_sub = []
            if i == j or incidence_matrix[i, j] == 0:
                continue
            if incidence_matrix[i, j] == 1:
                # find one pair, then create g_sub
                # if no old g_sub exists, add new graph, just like initialize
                if len(G.G_sub) == 0:
                    print("G.G_sub is equal to null")
                    G.initialize_new_G_sub(i, j, X)
                else:
                    # if already have old g_sub, check if it can insert into
                    # check if all the elements are pairwise intersected

                    # if current G_sub doesn't have G_sub with row i
                    for G_sub in G.G_sub:
                        # consider row as a new start sub_graph
                        isAddNewG_sub = True
                        # iterate all to judge if there is already have g_sub with row i
                        if i == G_sub.V[0]:
                            isAddNewG_sub = False
                            break;
                    if isAddNewG_sub == True:
                        G.initialize_new_G_sub(i, j, X)
                        break;

                    for G_sub in G.G_sub:
                        # add new vertex into pairwise check
                        print("G_sub.V", G_sub.V)
                        tem = G_sub.V
                        pair_C = list(combinations( np.append( np.array(G_sub.V), j), 2))
                        print("pari_c", pair_C)
                        checkRight = True
                        for pair in pair_C:
                            print(pair[0], pair[1])
                            # find one element don't pairwise intersect
                            # so the new one should create a g_sub with row i and column j
                            if incidence_matrix[pair[0], pair[1]] == 0:
                                G.initialize_new_G_sub(i, j, X)
                                checkRight = False
                                break
                        # add new vertex int old sub_graph
                        if checkRight == True:
                            G_sub.V.append(j)
                            G_sub.E.append([j, i])
                            G_sub.X.append( X[j] )
                        print("---", G_sub.V)
        # one row determines one overlap region graph
        # append the row one and don't need append E


    print(G.toarray())
    for i in G.G_sub:
        print("----different grap_sub---")
        print(i.toarray())

def test18():
    V = np.array([0,1,2,3,4])
    g1 = Overlap_G(V, None, None)
    v2 = np.array([0,2,3])
    g1.add_G(v2, None, None)
    print(g1.toarray())
    v2 = np.array([0,2,3])
    g1.add_G(v2, None, None)
    print(g1.toarray())
    g2 = g1.G_sub[0]
    print(g2.GCS())

# todo
# overlap_G: this stupid data structure let me meet a lot of questions

# test break
def test19():
    a = np.array([1,2,3])
    for j in a:
        print(j)
        for i in a:
            print(i)
            if i == 2:
                break
        print("---", j)

# judge whether one element in list or np.ndarray
def test20():
    a = [1,2,3,4]
    if 4 in a:
        print("it's in")

# how to append element dynamically in numpy from list to numpy
def test21():
    a = [1,2]
    b = np.append(np.array(a), [3,1])
    print(b)



'''
    innovative
    this is different with the algorithm in paper,
    but i think this is an right one.
    However, it only have a problem that how to solve one V that connects different vertex
'''
def generate_GCS(X, radius):
    X = X
    neigh = NearestNeighbors(radius=radius)
    neigh.fit(X)
    A = neigh.radius_neighbors_graph(X)

    incidence_matrix = A.toarray()
    row, col = np.shape(incidence_matrix)

    # initialize G
    G = Overlap_G(V = np.arange(len(X)), E = None, X = X)
    # create overlap region graph
    for i in range(row):
        # if one row is already used, we must find all overlap region graph of it
        # so j start from i to columns
        # find the overlap region graph in one row and only can find in one row
        for j in range(i+1, col):
            V = []
            E = []
            X_sub = []
            if i == j or incidence_matrix[i, j] == 0:
                continue
            if incidence_matrix[i, j] == 1:
                # find one pair, then create g_sub
                # if no old g_sub exists, add new graph, just like initialize
                if len(G.G_sub) == 0:
                    G.initialize_new_G_sub(i, j, X)
                else:
                    # if already have old g_sub, check if it can insert into
                    # check if all the elements are pairwise intersected

                    # if current G_sub doesn't have G_sub with row i
                    for G_sub in G.G_sub:
                        # consider row as a new start sub_graph
                        isAddNewG_sub = True
                        # iterate all to judge if there is already have g_sub with row i
                        if i == G_sub.V[0]:
                            isAddNewG_sub = False
                            break;
                    if isAddNewG_sub == True:
                        G.initialize_new_G_sub(i, j, X)
                        break;

                    for G_sub in G.G_sub:
                        # add new vertex into pairwise check
                        tem = G_sub.V
                        pair_C = list(combinations( np.append( np.array(G_sub.V), j), 2))
                        checkRight = True
                        for pair in pair_C:
                            print(pair[0], pair[1])
                            # find one element don't pairwise intersect
                            # so the new one should create a g_sub with row i and column j
                            if incidence_matrix[pair[0], pair[1]] == 0:
                                G.initialize_new_G_sub(i, j, X)
                                checkRight = False
                                break
                        # add new vertex int old sub_graph
                        if checkRight == True:
                            G_sub.V.append(j)
                            G_sub.E.append([j, i])
                            G_sub.X.append( X[j] )
        # one row determines one overlap region graph
        # append the row one and don't need append E


# algorithm3: disconnected region overlap graph
from Disconnected_Overlap_G import Disconnected_Overlap_G

def test22():
    # step1. build region overlap
    X = np.array( [[1, 2], [2, 2], [3, 2], [6, 2], [0, 2]] )
    print("X.shape", X.shape)
    rest_X = X
    rest_V = np.arange(len(X))

    #initialize the disconnected_graph
    disconnected_G = Disconnected_Overlap_G(None, None)
    while True:
        # the end condition
        if len(rest_V) == 0:
            break
        #initialize G_new
        G_new = Disconnected_Overlap_G(X=rest_X[0, :], V=rest_V[0])
        rest_X, rest_V = update_rest(G_new, rest_X, rest_V)
        while True:
            X_overlap, V_overlap = judge_overlap(G_new, rest_X, rest_V)
            # there are overlap points
            print("---overlap_X",X_overlap, V_overlap, "---")
            if len(V_overlap) != 0:
                # add V and X into current G
                G_new.V = np.append(G_new.V, V_overlap)
                G_new.V = np.append(G_new.X, X_overlap)
                rest_X, rest_V = update_rest(G_new, rest_X, rest_V)
            else:
                rest_X, rest_V = update_rest(G_new, rest_X, rest_V)
                disconnected_G.add(G_new)
                break

# update the rest
# delete the common element between G_new and rest_V in rest
def update_rest(G_new, rest_X, rest_V):
    index = np.where(G_new.V == rest_V)
    # delete by row
    rest_X = np.delete(rest_X, index, axis=0)
    rest_V = np.delete(rest_V, index, axis=0)
    return rest_X, rest_V

# judge if they are region overlap point
def judge_overlap(G_new, rest_X, rest_V):
    X_overlap = np.array([])
    V_overlap = np.array([])
    # create for loop forcely
    i = 0
    # because we cannot iterate an object with only one elment
    # then chu ci xia ce
    for tem in range(G_new.V):
        delta_result = delta(G_new.X[i: -1], rest_X[:, 0: -1])
        if (G_new.X.size == 1):
            sum_radius = G_new.X[-1] + rest_X[:, -1]
        else:
            sum_radius = G_new.X[i, -1] + rest_X[:, -1]
        index_inr = np.where(delta_result <= sum_radius)
        V_overlap = np.concatenate(V_overlap, rest_V[index_inr])
        i = i + 1
    # delete repeating elements in V_overlap
    _, i = np.unique(V_overlap, return_index=True)
    V_overlap[np.sort(i)]
    # X_overlap =
    return X_overlap, V_overlap

# a compressed code to simplify code
def update_overlap(X_overlap, V_overlap, rest_V, rest_X, index_inr):
    if rest_V is None or len(rest_V) == 0:
        V_overlap = rest_V[index_inr]
        X_overlap = rest_X[index_inr]


    # former_len = len(rest_V)
    # V_overlap = combine_np_1D(V_overlap, rest_V[index_inr])
    # # update X_overlap
    # tem_index = V_overlap[former_len: ]
    # X_overlap = rest_X[tem_index]


    return X_overlap, V_overlap

# retain the not repeating part of two different array into one array
def combine_np_1D(x, y):
    z = np.concatenate((x,y))
    _, i = np.unique(z, return_index=True)
    return z[np.sort(i)]



# how to use the power of numpy
def test24():
    a = np.array([1,2,3])
    print( a ** 2 )

# test the range of numpy
def test25():
    a = np.array([1,2,3])
    print(a[1:2])

# it can be operated plus and minus, even if the dimension is different
# and the dimension of result is as the same as the one with more dimension
def test26():
    a = np.array([1,2,3])
    b = np.array([1])
    print(b - a)

    a1 = np.array([[1,2],[2,3]])
    b1 = np.array([1])
    print(b1-a1)

# test the difference between incoming parameter and actual parameter
# when transfer a simple value like int, float, they are copied into this function
# when transfer a complex object like list, dictionary, they are quoted like pointer in C
# but when transfering numpy variable, it is just like int, it's copied
def test27(a):
    a = np.delete(a, 1)
    print(a)

# the difference between size and len
# size return the number of elements
# len returns the nubmer of row of the matrix
def test28():
    a = np.array([[1,2,3,4],[1,2,3,4]])
    print(a.size)
    print(len(a))

    b = np.array([])
    print(b.size)
    print(len(b))

    c = None
    print(c.size)
    print(len(c))

# a strange feature of for loop
def test29():
    for i in range(1):
        print(i)
    for i in 1:
        print(i)



import networkx as nx
def test30():
    g = nx.Graph()
    e1 = np.array([1,2])
    g.add_edge(1, 2)
    g.add_edge(0, 1)
    g.add_edge(2, 0)
    g.add_edge(3, 4)
    g.add_edge(5, 5)

    g.add_edge(3, 4)

    res = list(nx.connected_components(g))
    a = np.array(res)

def test31():
    X = np.array( [[1, 2], [2, 2], [3, 2], [6, 2], [0, 2], [100, 2]])
    disconnected_G = Util.build_overlap_region_graph(X)
    print(disconnected_G)



# how to remove value in np
# remove the target value
# you can also remove the fixed row and col
def test32():
    a = np.array([1,2,3])
    print(np.delete(a, np.where(a == 3)))

# two method to change the dimension of np
def test33():
    a = np.array((1,2,3))
    # flatten into one dimension
    a.flatten()
    # reshape into the dimension you want
    np.reshape(a, (-1, 2))

# todo
# notice1
# the value in numpy is float, not int

# test sum
def test34():
    a = np.array([[1,2,3],[4, 0,0]])
    print( np.sum(a, axis=1))

# how to use nx to find pairdef test35():
#     g = nx.Graph()
#     g.add_edge(0, 1)
#     g.add_edge(0, 2)
#     g.add_edge(1, 2)
#     g.add_edge(2, 10)
#     g.add_edge(3, 4)
#     g.add_edge(5, 5)
#
#     all_clique = list(nx.enumerate_all_cliques(g))
#     print("complete_g:\n{}".format(all_clique))
#     all_maximum_clique = list(nx.find_cliques(g))
#     print("all_maximum_clique:\n{}".format(all_maximum_clique))
#     maximum_clique = list(nx.make_max_clique_graph(g))
#     print("maximum_clique:\n{}".format(maximum_clique))
#     disc = list(nx.connected_components(g))
#     print("disc:\n{}".format(disc))
#     print("------")
#     for c in nx.connected_components(g):
#         print(c)
#         sub_g = g.subgraph(c)
#         all_clique = list(nx.find_cliques(sub_g))
#         all_clique_size = [len(i) for i in all_clique]
#         MCS = max(all_clique_size)
#         print(MCS)
#
#     S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
#     print("the disconnected component graph:\n".format(S))wise point for the aim of computing MCS

# transfer set into np
def test35():
    a = np.array(list({1,2,3}))
    print(a)

def generate_GCS(X, radius):
    X = X
    neigh = RadiusNeighborsClassifier.NearestNeighbors(radius = radius)
    neigh.fit(X)
    A = neigh.radius_neighbors_graph(X)

    incidence_matrix = A.toarray()
    row, col = np.shape(incidence_matrix)

    # initialize G
    G = Overlap_G(V = np.arange(len(X)), E = None, X = X)
    # create overlap region graph
    for i in range(row):
        # if one row is already used, we must find all overlap region graph of it
        # so j start from i to columns
        # find the overlap region graph in one row and only can find in one row
        for j in range(i+1, col):
            V = []
            E = []
            X_sub = []
            if i == j or incidence_matrix[i][j] == 0:
                continue
            if incidence_matrix[i][j] == 1:
                # find one pair, then create g_sub
                # if no old g_sub exists, add new graph, just like initialize
                if len(G.G_sub) == 0:
                    G.initialize_new_G_sub(i, j, X)
                else:
                    # if already have old g_sub, check if it can insert into
                    # check if all the elements are pairwise intersected

                    # if current G_sub doesn't have G_sub with row i
                    for G_sub in G.G_sub:
                        # consider row as a new start sub_graph
                        isAddNewG_sub = True
                        # iterate all to judge if there is already have g_sub with row i
                        if i == G_sub.V[0]:
                            isAddNewG_sub = False
                            break;
                    if isAddNewG_sub == True:
                        G.initialize_new_G_sub(i, j, X)
                        break;

                    for G_sub in G.G_sub:
                        # add new vertex into pairwise check
                        tem = G_sub.V
                        pair_C = list(combinations( np.append( np.array(G_sub.V), j), 2))
                        checkRight = True
                        for pair in pair_C:
                            print(pair[0], pair[1])
                            # find one element don't pairwise intersect
                            # so the new one should create a g_sub with row i and column j
                            if incidence_matrix[pair[0]][pair[1]] == 0:
                                G.initialize_new_G_sub(i, j, X)
                                checkRight = False
                                break
                        # add new vertex int old sub_graph
                        if checkRight == True:
                            G_sub.V.append(j)
                            G_sub.E.append([j, i])
                            G_sub.X.append( X[j] )
        # one row determines one overlap region graph
        # append the row one and don't need append E
    return G

# take function in knn
def test36():
    a = np.array([[1,2,3],[4,5,6]])
    # take row 1
    print(a.take(1 ,axis=0))
    # take col 1
    print(a.take(1, axis=1))

    mode = np.array([2,2, 0, 1, 1, 2, 2, 0, 0, 2])
    class_c = np.array([0, 1, 2])
    print(class_c.take(mode))

# find the index of max value in np
def test37():
    a = np.array([1,2,3,4,5])
    print(a.argmax())



if __name__ == "__main__":
    test21()
