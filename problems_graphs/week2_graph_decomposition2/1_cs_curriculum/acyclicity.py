#Uses python3

import sys


def dfs(u, adj, visited, stack):
    global is_dag
    visited[u] = True
    stack.append(u)
    for v in adj[u]:
        if v in stack:  
            is_dag = False
            return
        if not visited[v]:  dfs(v, adj, visited, stack)
    stack.pop()

def acyclic(adj, visited):
    global is_dag
    is_dag = True
    stack = []
    for v in range(len(adj)):
        if not visited[v]:
            dfs(v, adj, visited, stack)
            if not is_dag:  return 1
    return 0

if __name__ == '__main__':
    n, m = map(int, input().split())
    edges = []
    for i in range(m):
        edges.append(tuple(map(int, input().split())))
    adj = [[] for _ in range(n)]
    visited = [False for _ in range(n)]
    for (a, b) in edges:
        adj[a - 1].append(b - 1)

    result = acyclic(adj, visited)
    print(result)
