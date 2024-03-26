#Uses python3

import sys
import heapq

def distance(adj, s, t):
    #write your code here
    dist = [float('inf') for _ in range(len(adj))]
    dist[s] = 0
    pq = [(0, s)]
    while pq:
        _, u = heapq.heappop(pq)
        for v, wt in adj[u]:
            if dist[v] > dist[u] + wt:
                dist[v] = dist[u] + wt
                heapq.heappush(pq, (dist[v], v))
    if dist[t] != float('inf'): return dist[t]
    else:   return -1


if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    for a, b, w in edges:
        adj[a - 1].append((b - 1, w))
    s, t = map(int, input().split())
    s, t = s - 1, t - 1
    print(distance(adj, s, t))
