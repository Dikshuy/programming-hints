#Uses python3

import sys
from collections import deque


def shortest_paths(n, graph, adj, s, dist):
    #write your code here
    dist[s] = 0
    for _ in range(n-1):
        for u, v, w in graph:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
    
    neg = deque()
    for u, v, w in graph:
        if dist[v] > dist[u] + w:
            neg.append(v)
    
    visited = [False for _ in range(n+1)]
    while neg:
        u = neg.popleft()
        visited[u] = True
        dist[u] = -float('inf')
        for v in adj[u]:
            if not visited[v]:  neg.append(v)
    return dist


if __name__ == '__main__':
    n, m = map(int, input().split())
    graph = []
    for i in range(m):
        graph.append(tuple(map(int, input().split())))
    s = int(input())
    adj = [[] for _ in range(n+1)]
    for a, b, w in graph:
        adj[a].append(b)
    distance = [float('inf')] * (n+1)
    dist = shortest_paths(n, graph, adj, s, distance)
    for x in range(1, n+1):
        if dist[x] == float('inf'):
            print('*')
        elif dist[x] == -float('inf'):
            print('-')
        else:
            print(dist[x])

