#Uses python3

import sys
from collections import deque

def bipartite(adj):
    #write your code here
    dist = [float('inf') for _ in range(len(adj))]
    q = deque()
    q.append(0)
    dist[0] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == float('inf'):
                q.append(v)
                dist[v] = dist[u] + 1
            else:
                if (dist[u] - dist[v]) % 2 == 0:
                    return 0
    return 1

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    print(bipartite(adj))
