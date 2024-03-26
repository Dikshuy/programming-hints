#Uses python3

import sys

def dfs(adj, visited, u, post):
    #write your code here
    global clock
    visited[u] = True
    for v in adj[u]:
        if not visited[v]:  dfs(adj, visited, v, post)
    post[u] = clock
    clock += 1

def toposort(adj, visited):
    #write your code here
    post = [0]*len(adj)
    for v in range(len(adj)):
        if not visited[v]:
            dfs(adj, visited, v, post)
    post = list(enumerate(post[0:], start=0))
    post.sort(key=lambda x:x[1], reverse=True)
    return post

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    visited = [False for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
    clock = 1
    order = toposort(adj, visited)
    for x, time in order:
        print(x + 1, end=' ')

