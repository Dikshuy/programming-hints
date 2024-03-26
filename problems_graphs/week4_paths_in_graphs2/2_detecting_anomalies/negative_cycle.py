#Uses python3

import sys


def negative_cycle(n, graph):
    #write your code here
    dist = [10**19 for _ in range(n+1)]
    dist[1] = 0

    for _ in range(n-1):
        for u, v, w in graph:
            if dist[v] > dist[u] + w:
                dist[v] = dist[u] + w
    for u, v, w in graph:
        if dist[v] > dist[u] + w:
            return 1
    
    return 0


if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))

    print(negative_cycle(n, edges))
