#Uses python3

import sys
from collections import deque

def distance(adj, s, t):
    #write your code here
    dist = [-1 for _ in range(len(adj))]
    q = deque()
    q.append(s)
    dist[s] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                q.append(v)
                dist[v] = dist[u] + 1

    return dist[t]

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        adj[b - 1].append(a - 1)
    s, t = map(int, input().split())
    s, t = s - 1, t - 1
    print(distance(adj, s, t))
