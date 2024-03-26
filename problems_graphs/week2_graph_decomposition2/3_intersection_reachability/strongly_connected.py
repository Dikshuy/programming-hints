#Uses python3

import sys

sys.setrecursionlimit(200000)


def dfs(u, adj, visited, post):
    global clock
    visited[u] = True
    clock += 1
    for v in adj[u]:
        if not visited[v]:
            dfs(v, adj, visited, post)
    clock += 1
    post[u] = clock

def dfs_inv_adj(inv_adj):
    post = [0]*len(adj)
    visited = [False for _ in range(len(inv_adj))]
    for v in range(len(inv_adj)):
        if not visited[v]:
            dfs(v, inv_adj, visited, post)
    post = list(enumerate(post[0:], start=0))
    post.sort(key=lambda x:x[1], reverse=True)
    post_order = []
    for v, _ in post:
        post_order.append(v)
    return post_order

def number_of_strongly_connected_components(adj, inv_adj):
    global clock
    #write your code here
    scc = 0
    post = [0]*len(adj)
    post_order = dfs_inv_adj(inv_adj)
    visited = [False for _ in range(n)]
    for v in post_order:
        if not visited[v]:
            scc += 1
            dfs(v, adj, visited, post)
    return scc

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    inv_adj = [[] for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)
        inv_adj[b-1].append(a-1)
    clock = 1
    print(number_of_strongly_connected_components(adj, inv_adj))
