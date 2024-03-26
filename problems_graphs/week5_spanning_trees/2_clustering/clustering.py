#Uses python3
import sys
import math


def make_set(n):
    parent = [i for i in range(n)]
    rank = [0 for _ in range(n)]
    return parent, rank

def find(i, parent):
    if i != parent[i]:
        i = find(parent[i], parent)
    return parent[i]

def union(i, j, parent, rank):
    i_p = parent[i]
    j_p = parent[j]
    if i_p == j_p:
        return parent, rank
    else:
        if rank[i_p] > rank[j_p]:
            parent[j_p] = i_p
        else:
            parent[i_p] = j_p
            if rank[i_p] == rank[j_p]:  rank[j_p] += 1
        
        return parent, rank

def clustering(n, edges, k):
    #write your code here
    n_edges = 0
    parent, rank = make_set(n)
    edges.sort(key = lambda x: x[2])

    for u, v, w in edges:
        if find(u, parent) != find(v, parent):
            if n_edges == n - k: 
                return w
            else:
                n_edges += 1
                parent, rank = union(u, v, parent, rank)


if __name__ == '__main__':
    n = int(input())
    points = []
    for _ in range(n):
        a, b = map(int, input().split())
        points.append((a, b))

    edges = []
    for i in range(n):
        x0, y0 = points[i]
        for j in range(i+1, n):
            x1, y1 = points[j]
            dist = math.sqrt((x1-x0)**2 + (y1-y0)**2)
            edges.append((i, j, dist))

    k = int(input())
    print("{0:.9f}".format(clustering(n, edges, k)))
