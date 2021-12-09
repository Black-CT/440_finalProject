class Overlap_G():
    def __init__(self, V, E, X):
        self.V = V
        self.E = E
        self.G_sub = []
        self.X = X

    def add_G(self, V, E, X):
        G_new = Overlap_G(V, E, X)
        self.G_sub.append(G_new)

    # find the maximum GCS of this vertex
    def GCS(self, vertex):

        return len(self.V)

    def toarray(self):
        print("V: ", self.V)
        print("E: ", self.E)
        print("G_sub: ", self.G_sub)
        print("X: ", self.X)
        print("GCS: ", self.GCS())

    def initialize_new_G_sub(self,i ,j ,X):
        V = []
        E = []
        X_sub = []

        V.append(i)
        V.append(j)
        E.append([i, i])
        E.append([j, i])
        X_sub.append(X[i])
        X_sub.append(X[j])
        self.add_G(V, E, X_sub)

    def G_sub_toarray(self):
        n = 0
        for i in self.G_sub:
            print("----G_sub_{}----".format(n))
            print(i.toarray())
            n = n + 1
