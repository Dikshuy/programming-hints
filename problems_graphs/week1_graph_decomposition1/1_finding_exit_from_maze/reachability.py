#Uses python3

import sys

def reach(adj, x, y, visited):
    #write your code here
    visited[x] = True
    for v in adj[x]:
        if not visited[v]:
            reach(adj, v, y, visited)
    return 0

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    x, y = map(int, input().split())
    adj = [[] for _ in range(n)]
    x, y = x - 1, y - 1
    visited = [False for _ in range(len(adj))]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    reach(adj, x, y, visited)
    if visited[y]:  print(1)
    else:   print(0)
