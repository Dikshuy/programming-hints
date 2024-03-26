#Uses python3

import sys


def dfs(u, adj, visited):
    visited[u] = True
    for v in adj[u]:
        if not visited[v]:
            dfs(v, adj, visited)
    return

def number_of_components(adj, visited):
    cc = 0
    for v in range(len(adj)):
        if not visited[v]:
            cc += 1
            dfs(v, adj, visited)
    return cc

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    visited = [False for _ in range(len(adj))]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    print(number_of_components(adj, visited))
